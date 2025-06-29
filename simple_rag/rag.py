import asyncio
from openai import OpenAI
from antarys import Client


class SimpleRAG:
    def __init__(self):
        self.openai = OpenAI()
        self.antarys = None
        self.vectors = None

    async def init(self):
        self.antarys = Client(host="http://localhost:8080")
        await self.antarys.create_collection("docs_2", dimensions=1536)
        self.vectors = self.antarys.vector_operations("docs_2")

    def embed(self, text):
        return self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

    async def add(self, doc_id, content):
        await self.vectors.upsert([{
            "id": doc_id,
            "values": self.embed(content),
            "metadata": {"content": content}
        }])

    async def search(self, query, top_k=3):
        results = await self.vectors.query(
            vector=self.embed(query),
            top_k=top_k,
            include_metadata=True
        )
        return results["matches"]

    def generate(self, query, docs):
        context = "\n".join([doc["metadata"]["content"] for doc in docs])
        return self.openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}]
        ).choices[0].message.content

    async def query(self, question, verbose=False):
        docs = await self.search(question)
        answer = self.generate(question, docs)

        if verbose:
            print(f"Q: {question}")
            print(f"A: {answer}")
            for doc in docs:
                print(f"Source: {doc['id']} ({doc['score']:.3f})")

        return answer, docs


async def main():
    rag = SimpleRAG()
    await rag.init()

    await rag.add("AHNSW",
                  "Unlike tradtional sequential HNSW, we are using a different asynchronous approach to HNSW and eliminating thread locks with the help of architectural fine tuning. We will soon release more technical details on the Async HNSW algorithmic approach.")
    await rag.add("Antarys",
                  "Antarys is a multi-modal vector database and it uses the AHNSW algorithm to enhance it's performance to perform semantic searching based on similarity")

    await rag.query("what is Antarys?", verbose=True)


asyncio.run(main())
