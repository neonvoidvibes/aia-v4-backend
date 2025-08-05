# in scripts/verify_namespace_contents.py
import os
from dotenv import load_dotenv
import logging

# Add the project root to the path to allow importing from 'utils'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pinecone_utils import init_pinecone

# --- Configuration ---
INDEX_NAME = "river"
NAMESPACE_TO_CHECK = "river"
# ---------------------

def list_all_ids_paginated_and_sorted():
    """
    Connects to Pinecone, lists ALL vector IDs in a given namespace,
    correctly handling pagination for large namespaces, sorts the results,
    and prints them in a clean, readable format.
    """
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    load_dotenv()
    
    pc = init_pinecone()
    if not pc:
        logging.error("Failed to initialize Pinecone.")
        return

    try:
        index = pc.Index(INDEX_NAME)
        logging.info(f"Successfully connected to index '{INDEX_NAME}'.")
        
        stats = index.describe_index_stats()
        namespace_stats = stats.namespaces.get(NAMESPACE_TO_CHECK)
        if not namespace_stats:
            print(f"\nNamespace '{NAMESPACE_TO_CHECK}' was not found in the index stats.")
            return

        vector_count = namespace_stats.vector_count
        if vector_count == 0:
            print(f"\nNamespace '{NAMESPACE_TO_CHECK}' is empty.")
            return

        print(f"\nFound {vector_count} total vectors in namespace '{NAMESPACE_TO_CHECK}'. Fetching all IDs...")

        # --- CORRECTED PAGINATION LOGIC WITH VALID LIMIT ---
        all_ids = []
        # Use the maximum allowed limit of 99 for efficiency.
        # The iterator will handle making multiple calls until all IDs are fetched.
        for ids_batch in index.list(namespace=NAMESPACE_TO_CHECK, limit=99):
            all_ids.extend(ids_batch)

        all_ids.sort()

        print(f"\n--- Fetched and Sorted {len(all_ids)} Vector IDs ---")
        for i, vec_id in enumerate(all_ids):
            print(f"{i+1:03d}: {vec_id}")
        print("------------------")

        if len(all_ids) != vector_count:
            print(f"\n[WARNING] The number of fetched IDs ({len(all_ids)}) does not match the index stats ({vector_count}). The index may be updating.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    list_all_ids_paginated_and_sorted()