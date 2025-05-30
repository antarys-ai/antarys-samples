import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
import antarys
from tqdm.asyncio import tqdm
import numpy as np
import uuid
from datasets import load_dataset

MODE = 0
COLLECTION_NAME = "dbpedia_benchmark_100k"
VECTOR_SIZE = 1536
BATCH_SIZES = [1000]
QUERY_COUNTS = [100, 1000]
RESULTS_FILE = "bench_100k.txt"
SAMPLE_LIMIT = 100000
MAX_RETRIES = 3


async def load_huggingface_dataset(limit: int = SAMPLE_LIMIT) -> List[Dict[str, Any]]:
    print(f"Loading {limit} samples from Hugging Face dataset...")
    dataset = load_dataset("KShivendu/dbpedia-entities-openai-1M", split='train')

    samples = []
    for i, item in tqdm(enumerate(dataset), total=limit, desc="Processing dataset"):
        if i >= limit:
            break
        samples.append({
            "id": str(uuid.uuid4()),
            "values": item["openai"],
            "metadata": {
                "title": item["title"],
                "text": item["text"],
                "source": "dbpedia",
                "sample_id": i
            }
        })
    return samples


async def initialize(client):
    collections = await client.list_collections()
    if COLLECTION_NAME in collections:
        print(f"Deleting existing collection '{COLLECTION_NAME}'...")
        await client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    await client.create_collection(
        name=COLLECTION_NAME,
        dimensions=VECTOR_SIZE,
        enable_hnsw=True,
        shards=16,
        m=16,
        ef_construction=100
    )


async def benchmark_writes(client, samples: List[Dict[str, Any]]):
    results = []
    vector_ops = client.vector_operations(COLLECTION_NAME)

    for batch_size in BATCH_SIZES:
        print(f"\nBenchmarking writes with batch size {batch_size}")
        batch_times = []
        total_points = 0
        successful_batches = 0

        for i in tqdm(range(0, len(samples), batch_size), desc=f"Batch size {batch_size}"):
            batch = samples[i:i + batch_size]

            for attempt in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    result = await vector_ops.upsert(
                        batch,
                        batch_size=batch_size,
                        show_progress=False
                    )
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)
                    successful_batches += 1
                    total_points += result.get("upserted_count", len(batch))
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"  Failed after {MAX_RETRIES} attempts: {e}")
                        break
                    print(f"  Attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        if batch_times:
            avg_time = sum(batch_times) / len(batch_times)
            qps = batch_size / avg_time if avg_time > 0 else 0
            results.append({
                "operation": "write",
                "batch_size": batch_size,
                "total_points": total_points,
                "successful_batches": successful_batches,
                "avg_batch_time": avg_time,
                "qps": qps,
                "batch_times": batch_times[:100]
            })

    return results


async def benchmark_reads(client, samples: List[Dict[str, Any]]):
    results = []
    vector_ops = client.vector_operations(COLLECTION_NAME)

    for query_count in QUERY_COUNTS:
        print(f"\nBenchmarking {query_count} queries")
        query_times = []
        successful_queries = 0

        query_vectors = [sample["values"] for sample in samples[:query_count]]

        for i, vector in enumerate(tqdm(query_vectors, desc="Running queries")):
            try:
                if i % 4 == 0:
                    start_time = time.time()
                    await vector_ops.query(
                        vector=vector,
                        top_k=10,
                        include_metadata=True
                    )
                elif i % 4 == 1:
                    start_time = time.time()
                    await vector_ops.query(
                        vector=vector,
                        top_k=5,
                        include_metadata=True,
                        filter={"metadata.source": "dbpedia"}
                    )
                elif i % 4 == 2:
                    start_time = time.time()
                    await vector_ops.query(
                        vector=vector,
                        top_k=7,
                        include_metadata=True,
                        threshold=0.5
                    )
                else:
                    start_time = time.time()
                    await vector_ops.query(
                        vector=vector,
                        top_k=5,
                        include_metadata=True,
                        use_ann=True,
                        ef_search=200,
                        threshold=0.3
                    )

                query_time = time.time() - start_time
                query_times.append(query_time)
                successful_queries += 1
            except Exception as e:
                print(f"  Query {i + 1} failed: {e}")
                continue

        if query_times:
            avg_time = sum(query_times) / len(query_times)
            qps = successful_queries / sum(query_times) if query_times else 0
            results.append({
                "operation": "read",
                "query_count": query_count,
                "successful_queries": successful_queries,
                "avg_query_time": avg_time,
                "qps": qps,
                "query_times": query_times[:100]
            })

    return results


def save_results(results: List[Dict[str, Any]], filename: str):
    with open(filename, "w") as f:
        for result in results:
            if result["operation"] == "write":
                f.write(f"WRITE PERFORMANCE (batch size {result['batch_size']})\n")
                f.write(f"- Total points inserted: {result['total_points']}\n")
                f.write(f"- Successful batches: {result['successful_batches']}\n")
                f.write(f"- Average batch time: {result['avg_batch_time']:.4f}s\n")
                f.write(f"- Throughput: {result['qps']:.2f} vectors/sec\n")
                percentiles = np.percentile(result['batch_times'], [50, 90, 99])
                f.write(
                    f"- Percentiles: P50={percentiles[0]:.4f}s, P90={percentiles[1]:.4f}s, P99={percentiles[2]:.4f}s\n\n")
            else:
                f.write(f"READ PERFORMANCE ({result['query_count']} queries)\n")
                f.write(f"- Successful queries: {result['successful_queries']}\n")
                f.write(f"- Average query time: {result['avg_query_time']:.4f}s\n")
                f.write(f"- Throughput: {result['qps']:.2f} queries/sec\n")
                percentiles = np.percentile(result['query_times'], [50, 90, 99])
                f.write(
                    f"- Percentiles: P50={percentiles[0]:.4f}s, P90={percentiles[1]:.4f}s, P99={percentiles[2]:.4f}s\n\n")

        f.write("\nSUMMARY\n")
        f.write("=" * 70 + "\n")

        write_results = [r for r in results if r["operation"] == "write"]
        if write_results:
            best_write = max(write_results, key=lambda x: x["qps"])
            f.write(
                f"Best write performance: {best_write['qps']:.2f} vectors/sec (batch size {best_write['batch_size']})\n")

        read_results = [r for r in results if r["operation"] == "read"]
        if read_results:
            best_read = max(read_results, key=lambda x: x["qps"])
            f.write(f"Best read performance: {best_read['qps']:.2f} queries/sec ({best_read['query_count']} queries)\n")


async def main():
    samples = await load_huggingface_dataset()
    client = await antarys.create_client(
        host="http://localhost:8080",
        timeout=120,
        debug=False,
        use_http2=True,
        cache_size=1000
    )

    await initialize(client)

    all_results = []
    try:
        write_results = await benchmark_writes(client, samples)
        all_results.extend(write_results)

        read_results = await benchmark_reads(client, samples)
        all_results.extend(read_results)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
    finally:
        save_results(all_results, RESULTS_FILE)
        print(f"\nBenchmark complete! Results saved to {RESULTS_FILE}")
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
