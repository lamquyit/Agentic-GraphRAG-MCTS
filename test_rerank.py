import asyncio
import os
from pathlib import Path
import sys

# Ensure local modules are importable
sys.path.insert(0, str(Path.cwd()))

from rag_service import RAGService
from llm_provider import reset_usage_stats, get_usage_stats

async def test_rerank():
    print("🚀 Starting Rerank Test...")
    
    # Use a small storage name for testing
    storage_name = "rag_storage_test_2"
    root_path = Path.cwd()
    
    service = RAGService(root_path, storage_name=storage_name)
    print("Initializing RAG...")
    service.initialize()
    
    print(f"DEBUG: service.rag is {service.rag}")
    if service.rag:
        print(f"DEBUG: service.rag.lightrag is {service.rag.lightrag}")
    
    # Index two small texts to have multiple chunks to rerank
    print("Indexing test documents...")
    test_file1 = root_path / "test_doc1.txt"
    test_file1.write_text("Paris is the capital and most populous city of France, with an official estimated population of 2,102,650 residents as of 1 January 2023. France is a country in Western Europe.")
    
    test_file2 = root_path / "test_doc2.txt"
    test_file2.write_text("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. it is named after the engineer Gustave Eiffel, whose company designed and built the tower.")
    
    # Process documents
    service.process_document(str(test_file1), str(root_path / "output_for_report"))
    service.process_document(str(test_file2), str(root_path / "output_for_report"))
    
    question = "Where is the Eiffel Tower located and what is the capital of France?"
    print(f"Querying: {question}")
    
    reset_usage_stats()
    # Hybrid mode uses vector search which triggers reranking
    answer = await service.aquery(question, mode="hybrid")
    
    print(f"\nAnswer: {answer}")
    print(f"\nUsage Stats: {get_usage_stats()}")
    print("\n✅ Test complete. Check if 'Reranked ... using Qwen-based LLM reranker' appeared in logs.")

if __name__ == "__main__":
    asyncio.run(test_rerank())
