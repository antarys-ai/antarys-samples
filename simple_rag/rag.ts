import OpenAI from 'openai';
import { createClient } from "antarys";

class SimpleRAG {
	private openai: OpenAI;
	private antarys: any;
	private vectors: any;

	constructor() {
		this.openai = new OpenAI();
		this.antarys = null;
		this.vectors = null;
	}

	async init(): Promise<void> {
		this.antarys = createClient("http://localhost:8080");
		try {
			await this.antarys.createCollection({
				name: "docs",
				dimensions: 1536
			});
		} catch (error: any) {
			if (!error.message.includes('already exists')) {
				throw error;
			}
		}

		this.vectors = this.antarys.vectorOperations("docs");
	}

	async embed(text: string): Promise<number[]> {
		const response = await this.openai.embeddings.create({
			model: "text-embedding-3-small",
			input: text
		});
		return response.data[0].embedding;
	}

	async add(docId: string, content: string): Promise<void> {
		const embedding = await this.embed(content);
		await this.vectors.upsert([{
			id: docId,
			values: embedding,
			metadata: { content }
		}]);
	}

	async search(query: string, topK: number = 3): Promise<any[]> {
		const embedding = await this.embed(query);
		const results = await this.vectors.query({
			vector: embedding,
			topK,
			includeMetadata: true
		});
		return results.matches;
	}

	async generate(query: string, docs: any[]): Promise<string> {
		const context = docs.map(doc => doc.metadata.content).join("\n");
		const response = await this.openai.chat.completions.create({
			model: "gpt-4",
			messages: [{
				role: "user",
				content: `Context: ${context}\n\nQuestion: ${query}`
			}]
		});
		return response.choices[0].message.content || '';
	}

	async query(question: string, verbose: boolean = false): Promise<[string, any[]]> {
		const docs = await this.search(question);
		const answer = await this.generate(question, docs);

		if (verbose) {
			console.log(`Q: ${question}`);
			console.log(`A: ${answer}`);
			docs.forEach(doc => {
				console.log(`Source: ${doc.id} (${doc.score.toFixed(3)})`);
			});
		}

		return [answer, docs];
	}

	async close(): Promise<void> {
		if (this.antarys) {
			await this.antarys.close();
		}
	}
}

async function main() {
	const rag = new SimpleRAG();

	await rag.init();

	await rag.add("AHNSW",
		"Unlike tradtional sequential HNSW, we are using a different asynchronous approach to HNSW and eliminating thread locks with the help of architectural fine tuning. We will soon release more technical details on the Async HNSW algorithmic approach.");
	await rag.add("Antarys",
		"Antarys is a multi-modal vector database and it uses the AHNSW algorithm to enhance it's performance to perform semantic searching based on similarity");

	await rag.query("what is Antarys?", true);

	await rag.close();
}

main()
