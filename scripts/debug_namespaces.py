#!/usr/bin/env python3
"""Debug script to check namespace format"""
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
    namespaces_response = index.list_namespaces()
    
    print(f"Type of response: {type(namespaces_response)}")
    print(f"Response: {namespaces_response}")
    
    if hasattr(namespaces_response, 'namespaces'):
        all_namespaces = list(namespaces_response.namespaces)
    else:
        all_namespaces = list(namespaces_response)
    
    print(f"Type of all_namespaces: {type(all_namespaces)}")
    print(f"First item type: {type(all_namespaces[0]) if all_namespaces else 'empty'}")
    print(f"First item: {all_namespaces[0] if all_namespaces else 'empty'}")
    
    # Extract namespace names
    namespace_names = []
    for ns in all_namespaces:
        if hasattr(ns, 'name'):
            # NamespaceDescription object
            namespace_names.append(ns.name)
        elif isinstance(ns, dict):
            namespace_names.append(ns.get('name', str(ns)))
        else:
            namespace_names.append(str(ns))
    
    print(f"Namespace names: {namespace_names}")
    
    # Test filter for _test
    test_namespaces = [ns for ns in namespace_names if '_test' in ns.lower()]
    print(f"Test namespaces found: {test_namespaces}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()