import os
import logging
from dotenv import load_dotenv
from utils.pinecone_utils import init_pinecone
from utils.s3_utils import get_s3_client

# --- Configuration ---
# Set to False to perform the actual metadata updates in Pinecone.
# It is HIGHLY RECOMMENDED to run with DRY_RUN = True first to review the logs.
DRY_RUN = False

# The name of your shared Pinecone index.
INDEX_NAME = "river"

# List of agent names (which are namespaces in Pinecone) to fix.
# Add all agents that might have recordings without timestamps.
AGENTS_TO_FIX = ["river", "ikea", "cfl", "holistic", "_test", "nordicequation", "mobius", "sturebadet", "tpframtidenhr", "riveralmedalen", "neonvoid", "matfrid", "chat", "magictasks", "gotland2025", "infinitemg", "yggdrasil", "wlg", "newco", "tpframtiden", "tpframtidenhr", "samverket"] # <-- CONFIGURE THIS LIST
# ---------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def backfill_timestamps():
    """
    Finds vectors from 'recording' sources that are missing the 'created_at' timestamp,
    retrieves the LastModified date from S3, and updates the Pinecone metadata.
    """
    pc = init_pinecone()
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')

    if not all([pc, s3, aws_s3_bucket]):
        logging.error("Could not initialize Pinecone/S3 clients or find S3 bucket. Aborting.")
        return

    try:
        index = pc.Index(INDEX_NAME)
        logging.info(f"Successfully connected to index '{INDEX_NAME}'.")
    except Exception as e:
        logging.error(f"Could not connect to index '{INDEX_NAME}'. Error: {e}")
        return

    for agent_name in AGENTS_TO_FIX:
        namespace = agent_name
        logging.info(f"\n--- Processing Agent / Namespace: '{namespace}' ---")

        try:
            # We can't filter for a missing field, so we filter for the source
            # and then check each result for the missing timestamp.
            # We query with a dummy vector and a high top_k to fetch many candidates.
            # Pinecone's max top_k is 10,000, which should be sufficient.
            query_response = index.query(
                vector=[0]*1536,  # Dummy vector
                top_k=1000,       # Fetch up to 1000 recording vectors
                filter={"source": {"$eq": "recording"}},
                namespace=namespace,
                include_metadata=True
            )

            if not query_response.matches:
                logging.info(f"No vectors with source='recording' found in namespace '{namespace}'. Skipping.")
                continue

            logging.info(f"Found {len(query_response.matches)} candidate vectors from recordings.")
            
            updates_performed = 0
            for match in query_response.matches:
                metadata = match.metadata
                
                # The core logic: check if 'created_at' is missing.
                if 'created_at' not in metadata:
                    s3_key = metadata.get('s3_key')
                    if not s3_key:
                        logging.warning(f"Found vector {match.id} without 'created_at' but it also lacks an 's3_key'. Cannot fix. Metadata: {metadata}")
                        continue
                    
                    try:
                        # Use head_object for efficiency - it only gets metadata, not the file content.
                        s3_object_meta = s3.head_object(Bucket=aws_s3_bucket, Key=s3_key)
                        last_modified = s3_object_meta.get('LastModified')

                        if not last_modified:
                            logging.warning(f"Could not retrieve LastModified date for S3 key '{s3_key}' (vector {match.id}). Skipping.")
                            continue

                        # Convert datetime to ISO 8601 string format for consistency
                        created_at_iso = last_modified.isoformat()

                        logging.info(f"  [FIX] Vector '{match.id}' is missing timestamp. Found S3 LastModified: {created_at_iso}")

                        if not DRY_RUN:
                            index.update(
                                id=match.id,
                                set_metadata={'created_at': created_at_iso},
                                namespace=namespace
                            )
                            logging.info(f"    -> SUCCESS: Updated metadata in Pinecone.")
                            updates_performed += 1
                        else:
                            logging.info(f"    -> DRY RUN: Would have updated metadata with created_at: '{created_at_iso}'.")

                    except s3.exceptions.NoSuchKey:
                        logging.error(f"  [ERROR] S3 key '{s3_key}' for vector {match.id} not found in bucket. Cannot get timestamp.")
                    except Exception as e:
                        logging.error(f"  [ERROR] Failed to process vector {match.id} with S3 key '{s3_key}': {e}")
            
            if not DRY_RUN:
                logging.info(f"Finished processing for '{namespace}'. Performed {updates_performed} updates.")
            else:
                logging.info(f"Finished DRY RUN for '{namespace}'. Found {len([m for m in query_response.matches if 'created_at' not in m.metadata])} vectors that need updates.")

        except Exception as e:
            logging.error(f"An error occurred while processing namespace '{namespace}': {e}")

def main():
    setup_logging()
    load_dotenv()
    if DRY_RUN:
        print("="*40 + "\n    RUNNING IN DRY RUN MODE    \n" + "="*40)
        print("No data will be written. Review the output to ensure correctness.")
    else:
        print("="*40 + "\n    !!! LIVE BACKFILL MODE !!!    \n" + "="*40)
        print("This will write data to your Pinecone index. Ensure you have a backup.")
    backfill_timestamps()
    print("\nTimestamp backfill process finished.")

if __name__ == "__main__":
    main()