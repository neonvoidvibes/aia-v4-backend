#!/usr/bin/env python3
"""Check is_core_memory field preservation after migration"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "river"
pc = Pinecone()

try:
    index = pc.Index(INDEX_NAME)
    
    # Check a few different namespaces
    namespaces_to_check = ['_test', 'neonvoid', 'vibecloud']
    
    for namespace in namespaces_to_check:
        print(f"\n=== {namespace} namespace ===")
        
        # Get a few vectors to sample
        query_response = index.query(
            vector=[0.0] * 1536,
            top_k=3,
            namespace=namespace,
            include_metadata=True,
            include_values=False
        )
        
        for i, match in enumerate(query_response.matches):
            metadata = match.metadata or {}
            is_core = metadata.get('is_core_memory', 'NOT_SET')
            embed_model = metadata.get('embed_model', 'NOT_SET')
            
            print(f"Vector {i+1}:")
            print(f"  is_core_memory: {is_core}")
            print(f"  embed_model: {embed_model}")
            print(f"  id: {match.id[:50]}...")
            print()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()