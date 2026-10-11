from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
)

results = vectorstore.similarity_search("what makes Chroma DB different?", k=2)
for r in results:
    print(r.page_content)
    print("=" * 60)