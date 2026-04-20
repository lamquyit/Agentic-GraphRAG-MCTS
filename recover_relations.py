import asyncio
from pathlib import Path
import networkx as nx
from rag_service import RAGFactory
import sys

async def recover():
    print("Initializing RAG with empty relationships DB (first we rename the bad file)")
    bad_file = Path('/workspace/rag_update/codecuatui/rag_storage_hotpot/vdb_relationships.json')
    if bad_file.exists():
        bad_file.unlink() # Delete the corrupted file
        
    rag = RAGFactory.create_rag_sync(Path('/workspace/rag_update/codecuatui'), 'rag_storage_hotpot')
    res = await rag._ensure_lightrag_initialized()
    if not res.get('success'):
        print('Still failed:', res)
        return
        
    print("RAG loaded. Loading GraphML...")
    graph_path = Path('/workspace/rag_update/codecuatui/rag_storage_hotpot/graph_chunk_entity_relation.graphml')
    G = nx.read_graphml(graph_path)
    
    print(f"Graph loaded. {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # We need to recreate the relationship VDB.
    # Format according to LightRAG: src \t dest \n description
    # then embed it.
    edges_data = {}
    from lightrag.utils import compute_mdhash_id
    
    for src, tgt, edge_data in G.edges(data=True):
        desc = edge_data.get('description', '')
        if not desc:
            continue
        content = f"{src}\t{tgt}\n{desc}"
        id_str = compute_mdhash_id(f"{src}{tgt}", prefix="rel-")
        
        edges_data[id_str] = {
            "content": content,
            "src_id": src,
            "tgt_id": tgt,
            "weight": edge_data.get('weight', 1.0)
        }
    
    print(f"Prepared {len(edges_data)} relationships for VDB. Upserting...")
    
    # Batch upsert to vector DB
    await rag.lightrag.relationships_vdb.upsert(edges_data)
    
    # Finalize storages to write to disk
    await rag.lightrag.finalize_storages()
    print("Done! Relationship vectors successfully recovered.")

if __name__ == '__main__':
    asyncio.run(recover())
