# 04 · Intro to RAG

The retrieval half of a RAG pipeline over a single Medium post
(`mediumblog1.txt`).

- `ingestion.py`: load, split into ~1000-character chunks, embed with Ollama
  `nomic-embed-text` (768 dimensions), upsert into Pinecone
- `sanity_check.py`: similarity search against the index to confirm retrieval
  returns the right section

| Course used | Used here | Why |
|---|---|---|
| `OpenAIEmbeddings` | `OllamaEmbeddings` (`nomic-embed-text`) | Free, runs locally |
| `UnstructuredLoader` | `TextLoader` | Plain `.txt` input; `unstructured` is heavy |
| Pinecone | Pinecone (free Starter tier) | Index must be created with dimension 768, metric cosine |

Requires `ollama pull nomic-embed-text` and `PINECONE_API_KEY` / `INDEX_NAME`
in `.env`.

Next: the generation step (retrieved chunks + question → grounded answer).
