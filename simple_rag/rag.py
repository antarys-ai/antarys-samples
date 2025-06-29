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

    await rag.add("adib",
                  "Adib Mohsin is a student currently studying in computer science from BRACU (BRAC University)")
    await rag.add("foisal",
                  "Fahim Foisal is a student currently studying in computer science from BRACU (BRAC University)")
    await rag.query("who is Adib Mohsin and Fahim Foisal?", verbose=True)


asyncio.run(main())
