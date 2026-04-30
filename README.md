# Maintenance RAG Pipeline

A retrieval-augmented generation (RAG) system built from scratch in pure Python, no LangChain, no LlamaIndex. Designed for querying industrial safety and maintenance documents using natural language.

---

## What This Is

Most RAG tutorials hand you a framework and hide the internals. This project builds every component manually to demonstrate a clear understanding of what RAG actually does at each step:

- How text is parsed and segmented into meaningful units
- How meaning is encoded as vectors via embeddings
- How semantic similarity drives retrieval
- How retrieved context is structured into a prompt for generation

The domain: maintenance and manufacturing safety documentation is a real use case where accurate, grounded answers matter. Hallucination is not acceptable when the output is a safety instruction.

---

## Pipeline Architecture

```
parse() → chunk() → embedding() → retrieve() → generate()
```

Each stage is a discrete function with a single responsibility. They are deliberately decoupled so any stage can be swapped or improved independently.

---

## Design Decisions

### Parsing
The document is read as a single UTF-8 string using Python's built-in `open()`. No external parsing libraries are used for plain text. When PDF support is added, this is the only function that changes, the rest of the pipeline is format-agnostic.

### Chunking
Chunks are split on three or more consecutive newlines (`\n{3,}`) using Python's `re` module. This was chosen over fixed-size character chunking for one reason: the source documents are structured safety manuals where each section is a semantically complete unit. Splitting on structural boundaries preserves that meaning.

Fixed-size chunking was explicitly rejected here because it would split mid-sentence across section boundaries, destroying the context that makes a chunk useful for retrieval. For example, a numbered safety procedure split across two chunks loses the heading that identifies what procedure it belongs to.

Chunks are cleaned of empty or whitespace-only strings before being returned.

### Embedding
All chunks are embedded in a single API call using `text-embedding-ada-002`. The OpenAI embeddings API accepts a list as input and preserves order in its response, so chunks and their vectors are paired using `zip()` and stored as a list of dictionaries:

```python
{"chunk": "chunk text", "embedding": [0.123, ...]}
```

Batching the entire chunk list in one call was chosen over a per-chunk loop to minimize API round trips and latency.

### Retrieval
The user's query is embedded using the same model as the chunks. Cosine similarity is computed between the query vector and every chunk vector using NumPy:

```python
similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Cosine similarity was chosen over Euclidean distance because embedding vectors are high-dimensional and their magnitude carries less information than their direction. Cosine similarity measures the angle between vectors, which is a more reliable signal of semantic similarity in this space.

Results are sorted descending by score and the top N chunks are returned.

### Generation
Retrieved chunks are joined with a `---` separator and injected into the system prompt. The model is explicitly instructed to answer using only the provided context and to acknowledge when the answer is not present. This grounds the model's output in the document and prevents hallucination from general training knowledge.

---

## Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Embeddings | Azure OpenAI (`text-embedding-ada-002`) |
| Generation | Azure OpenAI (`gpt-4o`) |
| Vector math | NumPy |
| Secrets management | python-dotenv |

No vector database is used. Vectors are stored in memory as a list of dictionaries. This is intentional for this stage, it keeps the retrieval logic transparent and removes infrastructure dependencies. The tradeoff is that the index is rebuilt on every run, which is acceptable for a document corpus of this size.

---

## Project Structure

```
Maintenance_RAG/
├── main.py          # Full pipeline: parse, chunk, embed, retrieve, generate
├── sample.txt       # Sample maintenance and safety documentation
├── .env             # API keys (not committed)
├── .gitignore       # Excludes .env
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/Maintenance_RAG.git
cd Maintenance_RAG
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install openai numpy python-dotenv
```

**4. Add your credentials**

Create a `.env` file in the project root:
```
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
```

**5. Run**
```bash
python main.py
```

---

## Usage

Edit the `query` variable in the `__main__` block to ask any question about your documents:

```python
query = "what should I do before using a pallet jack?"
```

To change the source document, update `filepath`:

```python
filepath = "your_document.txt"
```

---

## Example Output

**Query:** `"What should I do before using a pallet jack?"`

**Response:**
> Prior to using any pallet jack — either manual or electric — you must inspect the equipment to ensure there was no damage from the prior shift and that you can safely operate it. Complete your plant's inspection form before use. Some plants use RealSPC for this checklist while others use a physical form — check with your department lead, trainer, or supervisor to confirm which method applies to your plant.

---

## Known Limitations and Next Steps

**In-memory index:** The vector index is rebuilt on every run. For larger document sets this becomes slow and expensive. The next step is persisting embeddings to a database (Supabase with pgvector) so the index only needs to be built once.

**No evaluation layer:** Retrieval quality is currently assessed manually by reading outputs. The next planned addition is a lightweight evaluation framework that scores the pipeline on a set of test questions with known answers, using an LLM-as-judge approach to measure answer faithfulness and retrieval relevance.

**Single document:** The pipeline is tested on one short document. Adding a large PDF will require revisiting the chunking strategy — specifically, whether section-boundary chunking remains appropriate or whether a hybrid approach (boundary-first, then size-limited) is needed.

**Chunk size variance:** Current chunks vary significantly in length depending on section size. This can skew retrieval toward longer chunks that contain more keywords. A future improvement is normalizing chunk length or adding a reranking step.

---

## What I Learned Building This

The most important insight from building RAG without a framework is understanding what the frameworks are actually abstracting. LangChain's `RecursiveCharacterTextSplitter`, for example, is solving the same problem as the chunking function here, but its default behavior is optimized for generic text and not structured documents. Knowing when to use the framework default and when to override it requires understanding what the default does, which is invisible if you start with the framework.

The second insight is that retrieval quality is the hardest part. The generation step is straightforward once you have good context. Getting the right chunks back for a given query — and knowing when retrieval is failing — requires evaluation infrastructure, not just manual inspection.