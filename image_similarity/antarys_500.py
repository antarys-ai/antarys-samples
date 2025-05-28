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
    def __init__(self, modelname: str):
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
    total_images = min(len(result_images) + 1, 501)
    grid_cols = 20
    grid_rows = math.ceil(total_images / grid_cols)

    plt.figure(figsize=(30, 30 * (grid_rows / grid_cols)))

    plt.subplot(grid_rows, grid_cols, 1)
    try:
        query_img = Image.open(query_image_path).resize((100, 100))
        plt.imshow(query_img)
        plt.title("QUERY", fontsize=6, fontweight='bold', color='red')
        plt.axis('off')

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
            spine.set_visible(True)
    except Exception as e:
        plt.text(0.5, 0.5, f"Error\n{str(e)[:15]}...",
                 ha='center', va='center', fontsize=4, transform=plt.gca().transAxes)
        plt.title("Query (Error)", fontsize=4)
        plt.axis('off')

    for i, img_path in enumerate(result_images[:500], start=2):
        plt.subplot(grid_rows, grid_cols, i)
        try:
            img = Image.open(img_path).resize((100, 100))
            plt.imshow(img)

            if similarities and len(similarities) >= i - 1:
                title = f"#{i - 1}\n{similarities[i - 2]:.2f}"
            else:
                title = f"#{i - 1}"

            plt.title(title, fontsize=5)
            plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f"Error\n{i - 1}",
                     ha='center', va='center', fontsize=4, transform=plt.gca().transAxes)
            plt.title(f"#{i - 1} (Error)", fontsize=4)
            plt.axis('off')

    total_results = len(result_images)
    plt.suptitle(f"Antarys Image Similarity Search Results (Top 500)\n"
                 f"Query time: {query_time:.4f} seconds | "
                 f"Total results: {total_results} | "
                 f"Showing: {min(500, total_results)} results",
                 fontsize=16, y=0.99)

    plt.tight_layout()

    filename = "similarity_search_results.png"
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Results saved as: {filepath}")
    plt.close()


def create_similarity_heatmap(similarities, top_n=500, output_dir=None):
    if not similarities:
        return

    top_similarities = similarities[:top_n]

    grid_cols = 20
    grid_rows = math.ceil(len(top_similarities) / grid_cols)

    padded_sims = top_similarities + [0] * (grid_cols * grid_rows - len(top_similarities))
    similarity_grid = np.array(padded_sims).reshape(grid_rows, grid_cols)

    plt.figure(figsize=(20, grid_rows))
    im = plt.imshow(similarity_grid, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Similarity Score')
    plt.title(f'Antarys Similarity Score Heatmap (Top {len(top_similarities)} Results)')
    plt.xlabel('Grid Position (X)')
    plt.ylabel('Grid Position (Y)')

    for i in range(grid_rows):
        for j in range(grid_cols):
            idx = i * grid_cols + j
            if idx < len(top_similarities) and top_similarities[idx] > 0.5:
                plt.text(j, i, f'{similarity_grid[i, j]:.2f}',
                         ha='center', va='center', fontsize=6, color='white')

    plt.tight_layout()

    filename = "similarity_heatmap.png"
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved as: {filepath}")
    plt.close()


def analyze_results(results, query_time):
    total_results = len(results.get("matches", [])) if results else 0

    if total_results == 0:
        print("No results found!")
        return [], []

    similarities = []
    result_paths = []

    for match in results["matches"]:
        similarities.append(match.get('score', 0))  # Antarys returns score directly
        result_paths.append(match["metadata"]["filename"])

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


async def create_performance_comparison(client, output_dir=None):
    try:
        stats = await client.get_performance_stats()
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        if "client" in stats:
            client_stats = stats["client"]
            metrics = ["avg_request_time_ms", "p95_request_time_ms", "p99_request_time_ms"]
            values = [client_stats.get(m, 0) for m in metrics]
            labels = ["Average", "P95", "P99"]

            plt.bar(labels, values, color=['green', 'orange', 'red'])
            plt.title('Request Response Times (ms)')
            plt.ylabel('Time (ms)')

        plt.subplot(2, 2, 2)
        if "vectors_processed" in stats and "queries_total" in stats:
            metrics = ["vectors_processed", "queries_total"]
            values = [stats.get(m, 0) for m in metrics]
            labels = ["Vectors Processed", "Queries Total"]

            plt.bar(labels, values, color=['blue', 'purple'])
            plt.title('Processing Statistics')
            plt.ylabel('Count')

        plt.subplot(2, 2, 3)
        if "client_cache" in stats:
            cache_stats = stats["client_cache"]
            hit_rate = cache_stats.get("hit_rate", 0) * 100
            miss_rate = 100 - hit_rate

            plt.pie([hit_rate, miss_rate], labels=['Cache Hits', 'Cache Misses'],
                    colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
            plt.title('Cache Performance')

        plt.subplot(2, 2, 4)
        capabilities = []
        capability_values = []

        if stats.get("gpu_enabled"):
            capabilities.append("GPU Enabled")
            capability_values.append(1)
        else:
            capabilities.append("GPU Disabled")
            capability_values.append(0)

        if stats.get("hnsw_enabled"):
            capabilities.append("HNSW Enabled")
            capability_values.append(1)
        else:
            capabilities.append("HNSW Disabled")
            capability_values.append(0)

        if "client" in stats and stats["client"].get("http2_enabled"):
            capabilities.append("HTTP/2 Enabled")
            capability_values.append(1)
        else:
            capabilities.append("HTTP/2 Disabled")
            capability_values.append(0)

        colors = ['green' if v else 'red' for v in capability_values]
        plt.bar(capabilities, capability_values, color=colors)
        plt.title('System Capabilities')
        plt.ylabel('Status')
        plt.ylim(0, 1.2)

        plt.tight_layout()

        filename = "performance_metrics.png"
        if output_dir:
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename

        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Performance metrics saved as: {filepath}")
        plt.close()

        return stats

    except Exception as e:
        print(f"Could not create performance comparison: {e}")
        return {}


async def main():
    timestamp = int(time.time())
    output_dir = f"antarys_500_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    client = await create_client(
        host="http://localhost:8080",
        use_http2=True,
        cache_size=5000,
        debug=True,
        thread_pool_size=16,
        connection_pool_size=40
    )

    collection_name = f"image_embeddings_500_{timestamp}"

    try:
        await client.create_collection(
            name=collection_name,
            dimensions=512,
            enable_hnsw=True,
            shards=32,
            m=64,
            ef_construction=600,
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

                            if inserted_count % 500 == 0:
                                print(f"Processed {inserted_count} images...")

                        except Exception as e:
                            print(f"Error processing {filepath}: {e}")

            print(f"Total images to insert: {len(records)}")

            if records:
                insert_result = await vector_ops.upsert(
                    records,
                    batch_size=2000,
                    show_progress=True,
                    parallel_workers=8
                )
                print(f"Insertion result: {insert_result}")

            await client.commit()
            print("Data committed to disk")

        query_image = "./test/Afghan_hound/n02088094_4261.JPEG"
        print(f"\nSearching for top 500 similar images to: {query_image}")

        query_embedding = extractor(query_image)

        start_time = time.time()

        search_results = await vector_ops.query(
            vector=query_embedding.tolist(),
            top_k=500,  # Increased to 500 results
            include_metadata=True,
            include_values=False,
            use_ann=True,
            ef_search=500,
            threshold=0.0
        )

        query_time = time.time() - start_time

        result_images, similarities = analyze_results(search_results, query_time)

        if result_images:
            display_results(query_image, result_images, query_time, similarities, output_dir)

            create_similarity_heatmap(similarities, top_n=500, output_dir=output_dir)

            performance_stats = await create_performance_comparison(client, output_dir)

            print(f"Images displayed: {min(500, len(result_images))}")
            print(f"Best similarity score: {max(similarities):.4f}")
            print(f"Worst similarity score: {min(similarities):.4f}")
            print(f"Standard deviation: {np.std(similarities):.4f}")

            plt.figure(figsize=(12, 6))
            plt.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.title('Antarys Similarity Score Distribution (Top 500 Results)')
            plt.grid(True, alpha=0.3)

            dist_filename = os.path.join(output_dir, "similarity_distribution.png")
            plt.savefig(dist_filename, dpi=150, bbox_inches='tight')
            print(f"Distribution plot saved as: {dist_filename}")
            plt.close()

            summary_file = os.path.join(output_dir, "search_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Antarys Image Similarity Search Summary (Top 500)\n")
                f.write(f"===============================================\n\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Collection: {collection_name}\n")
                f.write(f"Query Image: {query_image}\n")
                f.write(f"Query Execution Time: {query_time:.4f} seconds\n")
                f.write(f"Total Results Found: {len(result_images)}\n")
                f.write(f"Images Displayed: {min(500, len(result_images))}\n\n")

                f.write(f"Similarity Statistics:\n")
                f.write(f"- Best Score: {max(similarities):.4f}\n")
                f.write(f"- Worst Score: {min(similarities):.4f}\n")
                f.write(f"- Average Score: {np.mean(similarities):.4f}\n")
                f.write(f"- Standard Deviation: {np.std(similarities):.4f}\n\n")

                f.write(f"Antarys Configuration:\n")
                f.write(f"- Collection Dimensions: 512\n")
                f.write(f"- HNSW Enabled: True\n")
                f.write(f"- Shards: 32\n")
                f.write(f"- M Parameter: 64\n")
                f.write(f"- EF Construction: 600\n")
                f.write(f"- EF Search: 500\n\n")

                if performance_stats:
                    f.write(f"Performance Statistics:\n")
                    if "client" in performance_stats:
                        client_stats = performance_stats["client"]
                        f.write(f"- Average Request Time: {client_stats.get('avg_request_time_ms', 0):.2f} ms\n")
                        f.write(f"- P95 Request Time: {client_stats.get('p95_request_time_ms', 0):.2f} ms\n")
                        f.write(f"- Total Requests: {client_stats.get('total_requests', 0)}\n")
                        f.write(f"- HTTP/2 Enabled: {client_stats.get('http2_enabled', False)}\n")
                        f.write(f"- Thread Pool Size: {client_stats.get('thread_pool_size', 0)}\n")
                        f.write(f"- Connection Pool Size: {client_stats.get('connection_pool_size', 0)}\n")

                    if "client_cache" in performance_stats:
                        cache_stats = performance_stats["client_cache"]
                        f.write(f"- Cache Hit Rate: {cache_stats.get('hit_rate', 0):.2%}\n")
                        f.write(f"- Cache Size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}\n")

                    f.write(f"- GPU Enabled: {performance_stats.get('gpu_enabled', False)}\n")
                    f.write(f"- HNSW Enabled: {performance_stats.get('hnsw_enabled', False)}\n")
                    f.write(f"- Optimization Level: {performance_stats.get('optimization_level', 0)}\n")

                f.write(f"\nTop 50 Results:\n")
                for i, (path, sim) in enumerate(zip(result_images[:50], similarities[:50])):
                    f.write(f"  {i + 1:3d}. {sim:.4f} - {path}\n")
                f.write(f"\n...\n\nBottom 50 Results:\n")
                for i, (path, sim) in enumerate(zip(result_images[-50:], similarities[-50:])):
                    f.write(f"  {len(result_images) - 49 + i:3d}. {sim:.4f} - {path}\n")

            print(f"Summary saved as: {summary_file}")

            config_file = os.path.join(output_dir, "experiment_config.txt")
            with open(config_file, 'w') as f:
                f.write(f"Antarys Experiment Configuration (500-image)\n")
                f.write(f"=========================================\n\n")
                f.write(f"Model: resnet34\n")
                f.write(f"Vector Dimensions: 512\n")
                f.write(f"Collection Name: {collection_name}\n")
                f.write(f"Batch Size: 2000\n")
                f.write(f"Parallel Workers: 8\n")
                f.write(f"Top K Results: 500\n")
                f.write(f"Use ANN: True\n")
                f.write(f"EF Search: 500\n")
                f.write(f"Client Settings:\n")
                f.write(f"- HTTP/2: True\n")
                f.write(f"- Cache Size: 5000\n")
                f.write(f"- Thread Pool: 16\n")
                f.write(f"- Connection Pool: 40\n")

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
