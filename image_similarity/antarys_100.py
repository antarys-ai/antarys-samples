import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import asyncio
import os
import matplotlib.pyplot as plt
from antarys import create_client
import time
import uuid
import numpy as np
import math


class FeatureExtractor:
    def __init__(self, modelname):
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        input_image = Image.open(imagepath).convert("RGB")
        input_image = self.preprocess(input_image)
        input_tensor = input_image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)

        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


def display_results(query_image_path, result_images, query_time, similarities=None, output_dir=None):
    total_images = min(len(result_images) + 1, 101)
    grid_size = math.ceil(math.sqrt(total_images))

    plt.figure(figsize=(20, 20))

    plt.subplot(grid_size, grid_size, 1)
    try:
        query_img = Image.open(query_image_path).resize((150, 150))
        plt.imshow(query_img)
        plt.title("QUERY IMAGE", fontsize=8, fontweight='bold', color='red')
        plt.axis('off')

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
            spine.set_visible(True)
    except Exception as e:
        plt.text(0.5, 0.5, f"Error loading\nquery image:\n{str(e)}",
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Query Image (Error)", fontsize=8)
        plt.axis('off')

    for i, img_path in enumerate(result_images[:100], start=2):
        plt.subplot(grid_size, grid_size, i)
        try:
            img = Image.open(img_path).resize((150, 150))
            plt.imshow(img)

            if similarities and len(similarities) >= i - 1:
                title = f"#{i - 1}\nSim: {similarities[i - 2]:.3f}"
            else:
                title = f"Result #{i - 1}"

            plt.title(title, fontsize=6)
            plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f"Error loading\nimage {i - 1}",
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f"Result #{i - 1} (Error)", fontsize=6)
            plt.axis('off')

    total_results = len(result_images)
    plt.suptitle(f"Antarys Image Similarity Search Results\n"
                 f"Query time: {query_time:.4f} seconds | "
                 f"Total results: {total_results} | "
                 f"Showing: {min(100, total_results)} results",
                 fontsize=16, y=0.98)

    plt.tight_layout()

    filename = "similarity_search_results.png"
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Results saved as: {filepath}")
    plt.show()


def create_similarity_heatmap(similarities, top_n=100, output_dir=None):
    if not similarities:
        return

    top_similarities = similarities[:top_n]

    grid_size = math.ceil(math.sqrt(len(top_similarities)))

    padded_sims = top_similarities + [0] * (grid_size * grid_size - len(top_similarities))
    similarity_grid = np.array(padded_sims).reshape(grid_size, grid_size)

    plt.figure(figsize=(12, 10))
    im = plt.imshow(similarity_grid, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Similarity Score')
    plt.title(f'Antarys Similarity Score Heatmap (Top {len(top_similarities)} Results)')
    plt.xlabel('Grid Position (X)')
    plt.ylabel('Grid Position (Y)')

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(top_similarities):
                plt.text(j, i, f'{similarity_grid[i, j]:.3f}',
                         ha='center', va='center', fontsize=8, color='white')

    plt.tight_layout()

    # Save in the specified directory
    filename = "similarity_heatmap.png"
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved as: {filepath}")
    plt.show()


def analyze_results(results, query_time):
    total_results = len(results.get("matches", [])) if results else 0

    if total_results == 0:
        print("No results found!")
        return [], []

    similarities = []
    result_paths = []

    for match in results["matches"]:
        similarities.append(match.get('score', 0))
        result_paths.append(match["metadata"]["filename"])

    print(f"\n=== Antarys SEARCH RESULTS ANALYSIS ===")
    print(f"Query execution time: {query_time:.4f} seconds")
    print(f"Total results found: {total_results}")
    print(f"Top 10 similarity scores:")

    for i, sim in enumerate(similarities[:10]):
        print(f"  Rank {i + 1}: {sim:.4f}")

    if len(similarities) > 10:
        print(f"  ...")
        print(f"  Rank {len(similarities)}: {similarities[-1]:.4f}")

    print(f"Average similarity: {np.mean(similarities):.4f}")
    print(f"Similarity range: {min(similarities):.4f} - {max(similarities):.4f}")

    return result_paths, similarities


async def main():
    timestamp = int(time.time())
    output_dir = f"antarys_100_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    client = await create_client(
        host="http://localhost:8080",
        use_http2=True,
        cache_size=1000,
        debug=True,
        thread_pool_size=8,
        connection_pool_size=20
    )

    collection_name = f"image_embeddings_{timestamp}"

    try:
        await client.create_collection(
            name=collection_name,
            dimensions=512,
            enable_hnsw=True,
            shards=16,
            m=32,
            ef_construction=400,
        )
        print(f"Created collection: {collection_name}")

        vector_ops = client.vector_operations(collection_name)
        extractor = FeatureExtractor("resnet34")

        root = "./train"
        insert = True

        if insert:
            print("Extracting features and inserting images into Antarys...")
            records = []
            inserted_count = 0

            for dirpath, foldername, filenames in os.walk(root):
                for filename in filenames:
                    if filename.endswith(".JPEG"):
                        filepath = os.path.join(dirpath, filename)
                        try:
                            image_embedding = extractor(filepath)
                            record = {
                                "id": str(uuid.uuid4()),
                                "values": image_embedding.tolist(),
                                "metadata": {"filename": filepath}
                            }
                            records.append(record)
                            inserted_count += 1

                            if inserted_count % 100 == 0:
                                print(f"Processed {inserted_count} images...")

                        except Exception as e:
                            print(f"Error processing {filepath}: {e}")

            print(f"Total images to insert: {len(records)}")

            if records:
                insert_result = await vector_ops.upsert(
                    records,
                    batch_size=1000,
                    show_progress=True,
                    parallel_workers=4
                )
                print(f"Insertion result: {insert_result}")

            await client.commit()
            print("Data committed to disk")

        query_image = "./test/Afghan_hound/n02088094_4261.JPEG"
        print(f"\nSearching for top 100 similar images to: {query_image}")

        query_embedding = extractor(query_image)

        start_time = time.time()

        search_results = await vector_ops.query(
            vector=query_embedding.tolist(),
            top_k=100,
            include_metadata=True,
            include_values=False,
            use_ann=True,
            threshold=0.0
        )

        query_time = time.time() - start_time

        result_images, similarities = analyze_results(search_results, query_time)

        if result_images:
            display_results(query_image, result_images, query_time, similarities, output_dir)

            create_similarity_heatmap(similarities, top_n=100, output_dir=output_dir)

            print(f"Images displayed: {min(100, len(result_images))}")
            print(f"Best similarity score: {max(similarities):.4f}")
            print(f"Worst similarity score: {min(similarities):.4f}")
            print(f"Standard deviation: {np.std(similarities):.4f}")

            plt.figure(figsize=(10, 6))
            plt.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.title('Antarys Similarity Score Distribution (Top 100 Results)')
            plt.grid(True, alpha=0.3)

            dist_filename = os.path.join(output_dir, "similarity_distribution.png")
            plt.savefig(dist_filename, dpi=150, bbox_inches='tight')
            print(f"Distribution plot saved as: {dist_filename}")
            plt.show()

            summary_file = os.path.join(output_dir, "search_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"=====================================\n\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Collection: {collection_name}\n")
                f.write(f"Query Image: {query_image}\n")
                f.write(f"Query Execution Time: {query_time:.4f} seconds\n")
                f.write(f"Total Results Found: {len(result_images)}\n")
                f.write(f"Images Displayed: {min(100, len(result_images))}\n\n")

                f.write(f"Similarity Statistics:\n")
                f.write(f"- Best Score: {max(similarities):.4f}\n")
                f.write(f"- Worst Score: {min(similarities):.4f}\n")
                f.write(f"- Average Score: {np.mean(similarities):.4f}\n")
                f.write(f"- Standard Deviation: {np.std(similarities):.4f}\n\n")

                f.write(f"Antarys Configuration:\n")
                f.write(f"- Collection Dimensions: 512\n")
                f.write(f"- HNSW Enabled: True\n")
                f.write(f"- Shards: 16\n")
                f.write(f"- M Parameter: 32\n")
                f.write(f"- EF Construction: 400\n")
                f.write(f"- EF Search: 200\n\n")

                f.write(f"\nResults:\n")
                for i, (path, sim) in enumerate(zip(result_images, similarities)):
                    f.write(f"  {i + 1:2d}. {sim:.4f} - {path}\n")

            print(f"Summary saved as: {summary_file}")

            config_file = os.path.join(output_dir, "experiment_config.txt")
            with open(config_file, 'w') as f:
                f.write(f"Antarys Experiment Configuration\n")
                f.write(f"==============================\n\n")
                f.write(f"Model: resnet34\n")
                f.write(f"Vector Dimensions: 512\n")
                f.write(f"Collection Name: {collection_name}\n")
                f.write(f"Batch Size: 1000\n")
                f.write(f"Parallel Workers: 4\n")
                f.write(f"Top K Results: 100\n")
                f.write(f"Use ANN: True\n")
                f.write(f"EF Search: 200\n")
                f.write(f"Client Settings:\n")
                f.write(f"- HTTP/2: True\n")
                f.write(f"- Cache Size: 1000\n")
                f.write(f"- Thread Pool: 8\n")
                f.write(f"- Connection Pool: 20\n")

            print(f"Configuration saved as: {config_file}")
            print(f"\nAll outputs saved in directory: {output_dir}")

        else:
            print("No results found to display!")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await client.close()
        print("Antarys client closed successfully")


if __name__ == "__main__":
    asyncio.run(main())
