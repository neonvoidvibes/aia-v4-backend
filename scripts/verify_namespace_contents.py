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

def list_all_ids_paginated():
    """
    Connects to Pinecone and lists ALL vector IDs in a given namespace,
    correctly handling pagination for large namespaces.
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
        
        # First, get the total vector count from stats for confirmation
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

        # --- CORRECT PAGINATION LOGIC ---
        # The `index.list()` method in the Pinecone client returns an iterator
        # that automatically handles the pagination tokens behind the scenes.
        # We can simply loop through it to get all IDs, regardless of the total count.
        all_ids = []
        # The `limit` parameter here controls the batch size for each underlying API call.
        for vec_id in index.list(namespace=NAMESPACE_TO_CHECK, limit=100):
            all_ids.append(vec_id)

        print(f"\n--- Fetched {len(all_ids)} Vector IDs ---")
        for i, vec_id in enumerate(all_ids):
            print(f"{i+1}: {vec_id}")
        print("------------------")

        if len(all_ids) != vector_count:
            print(f"\n[WARNING] The number of fetched IDs ({len(all_ids)}) does not match the index stats ({vector_count}). The index may be updating.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    list_all_ids_paginated()