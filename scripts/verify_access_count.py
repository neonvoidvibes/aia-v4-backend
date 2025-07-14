import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project's root directory to the Python path to allow importing from 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pinecone_utils import init_pinecone

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def verify_access_count(index_name: str, namespace: str, vector_id: str):
    """
    Connects to Pinecone and fetches the metadata for a specific vector
    to check its access_count.
    """
    logger.info(f"Initializing connection to Pinecone index '{index_name}'...")
    pc = init_pinecone()
    if not pc:
        logger.error("Failed to initialize Pinecone. Check API key and environment settings.")
        return

    try:
        if index_name not in pc.list_indexes().names():
            logger.error(f"Index '{index_name}' does not exist in the current Pinecone environment.")
            return
        index = pc.Index(index_name)
        logger.info(f"Successfully connected to index '{index_name}'.")

        logger.info(f"Fetching vector '{vector_id}' from namespace '{namespace}'...")
        fetch_response = index.fetch(ids=[vector_id], namespace=namespace)

        vectors = fetch_response.get('vectors', {})
        if not vectors or vector_id not in vectors:
            logger.error(f"Vector with ID '{vector_id}' not found in namespace '{namespace}'.")
            return

        vector_data = vectors.get(vector_id)
        if not vector_data:
            logger.error(f"Could not retrieve data for vector ID '{vector_id}'.")
            return

        metadata = vector_data.get('metadata', {})
        access_count = metadata.get('access_count', 0)
        content = metadata.get('content', 'N/A')

        print("\n--- Verification Result ---")
        print(f"Vector ID:      {vector_id}")
        print(f"Namespace:      {namespace}")
        print(f"Access Count:   {access_count}")
        print(f"Content Snippet: '{content[:150]}...'")
        print("-------------------------\n")

    except Exception as e:
        logger.error(f"An error occurred during verification: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the access_count of a specific vector in Pinecone.")
    parser.add_argument("--index", type=str, default="river", help="The name of the Pinecone index to check.")
    parser.add_argument("--agent", required=True, help="The agent name, which is used as the Pinecone namespace.")
    parser.add_argument("--id", required=True, help="The specific vector ID to verify.")

    args = parser.parse_args()

    verify_access_count(index_name=args.index, namespace=args.agent, vector_id=args.id)
