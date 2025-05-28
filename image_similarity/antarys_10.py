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

GPU = True


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


def display_results(query_image_path, result_images, query_time=None):
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 6, 1)
    query_img = Image.open(query_image_path).resize((150, 150))
    plt.imshow(query_img)
    title = "Query Image"
    if query_time:
        title += f"\n({query_time:.4f}s)"
    plt.title(title)
    plt.axis('off')

    for i, img_path in enumerate(result_images, start=2):
        plt.subplot(2, 6, i)
        img = Image.open(img_path).resize((150, 150))
        plt.imshow(img)
        plt.title(f"Result {i - 1}")
        plt.axis('off')

    plt.tight_layout()

    if GPU:
        plt.savefig(f"result_antarys_{int(time.time())}_gpu.png")
    else:
        plt.savefig(f"result_antarys_{int(time.time())}_cpu.png")

    plt.show()


async def main():
    client = await create_client(
        host="http://localhost:8080",
        use_http2=True,
        cache_size=1000
    )

    collection_name = f"image_embeddings_{int(time.time())}"
    await client.create_collection(
        name=collection_name,
        dimensions=512,
    )

    vector_ops = client.vector_operations(collection_name)
    extractor = FeatureExtractor("resnet34")

    root = "./train"
    insert = True
    if insert:
        records = []
        for dirpath, foldername, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".JPEG"):
                    filepath = os.path.join(dirpath, filename)
                    image_embedding = extractor(filepath)
                    record = {
                        "id": str(uuid.uuid4()),
                        "values": image_embedding.tolist(),
                        "metadata": {"filename": filepath}
                    }
                    records.append(record)

        await vector_ops.upsert(
            records,
            batch_size=1000,
            show_progress=True
        )

        await client.commit()

    query_image = "./test/Afghan_hound/n02088094_4261.JPEG"
    query_embedding = extractor(query_image)

    start_time = time.time()

    search_results = await vector_ops.query(
        vector=query_embedding.tolist(),
        include_metadata=True,
        include_values=False,
        use_ann=True,
    )

    query_time = time.time() - start_time
    print(f"Query executed in {query_time:.4f} seconds")

    result_images = []
    for result in search_results["matches"]:
        result_images.append(result["metadata"]["filename"])

    display_results(query_image, result_images, query_time)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
