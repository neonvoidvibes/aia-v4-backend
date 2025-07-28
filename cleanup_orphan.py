# cleanup_orphan.py
import os
import logging
from dotenv import load_dotenv
from utils.pinecone_utils import init_pinecone

# --- Configuration ---
INDEX_NAME = "river"
NAMESPACE_TO_CLEAN = "chat"
VECTOR_ID_TO_DELETE = "recording_D20250715-T144902_uID-f8e3e957-5872-406f-83b6-121af5ab4dc9_oID-river_aID-chat_sID-818c178f827447b9955dff1e4625ed25.txt_0"
# ---------------------

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def delete_orphan_vector():
    """Deletes a specific orphaned vector from a Pinecone namespace."""
    pc = init_pinecone()
    if not pc:
        logging.error("Could not initialize Pinecone client. Aborting.")
        return

    try:
        index = pc.Index(INDEX_NAME)
        logging.info(f"Successfully connected to index '{INDEX_NAME}'.")
    except Exception as e:
        logging.error(f"Could not connect to index '{INDEX_NAME}'. Error: {e}")
        return

    logging.info(f"Attempting to delete vector '{VECTOR_ID_TO_DELETE}' from namespace '{NAMESPACE_TO_CLEAN}'...")
    try:
        delete_response = index.delete(ids=[VECTOR_ID_TO_DELETE], namespace=NAMESPACE_TO_CLEAN)
        logging.info(f"Pinecone delete operation completed. Response: {delete_response}")
        print("\nSUCCESS: Orphaned vector has been deleted.")
    except Exception as e:
        logging.error(f"An error occurred during deletion: {e}")
        print(f"\nERROR: Could not delete vector. Please check the logs.")

if __name__ == "__main__":
    setup_logging()
    load_dotenv()
    delete_orphan_vector()