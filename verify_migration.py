import os
from dotenv import load_dotenv
import logging
from utils.pinecone_utils import init_pinecone
from utils.supabase_client import get_supabase_client

# --- Configuration ---
AGENT_NAME_TO_VERIFY = "neonvoid"  # An agent that was migrated
NUM_RECORDS_TO_CHECK = 5           # How many records to check
# ---------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def verify_migration():
    """
    Verifies that Supabase records can be found in the new Pinecone namespace.
    """
    setup_logging()
    load_dotenv()

    # 1. Initialize clients
    supabase = get_supabase_client()
    pc = init_pinecone()

    if not supabase or not pc:
        logging.error("Could not initialize Supabase or Pinecone client. Aborting.")
        return

    try:
        river_index = pc.Index("river")
        logging.info("Successfully connected to 'river' index.")
    except Exception as e:
        logging.error(f"Could not connect to 'river' index. Please ensure it exists. Error: {e}")
        return

    # 2. Fetch recent logs from Supabase for the specified agent
    logging.info(f"Fetching last {NUM_RECORDS_TO_CHECK} memory logs for agent '{AGENT_NAME_TO_VERIFY}' from Supabase...")
    try:
        response = supabase.table("agent_memory_logs") \
            .select("id, source_identifier, created_at") \
            .eq("agent_name", AGENT_NAME_TO_VERIFY) \
            .order("created_at", desc=True) \
            .limit(NUM_RECORDS_TO_CHECK) \
            .execute()

        if not response.data:
            logging.warning(f"No memory logs found for agent '{AGENT_NAME_TO_VERIFY}' in Supabase. Cannot verify.")
            return
        
        logs_to_check = response.data
        logging.info(f"Found {len(logs_to_check)} logs to check.")

    except Exception as e:
        logging.error(f"Error fetching logs from Supabase: {e}")
        return

    # 3. For each log, verify its existence in the correct Pinecone namespace
    all_verified = True
    for log in logs_to_check:
        source_id = log.get('source_identifier')
        log_id = log.get('id')
        logging.info(f"\n--- Verifying Supabase Log ID: {log_id} (Source: {source_id}) ---")

        if not source_id:
            logging.warning(f"Log ID {log_id} has no source_identifier. Skipping.")
            continue

        try:
            # Query Pinecone using the logic from the updated application
            # We query a dummy vector because we only care about the metadata filter
            query_response = river_index.query(
                vector=[0]*1536,  # A zero vector is fine for a metadata-only check
                top_k=1,
                filter={"source_identifier": {"$eq": source_id}},
                namespace=AGENT_NAME_TO_VERIFY
            )

            if query_response.matches:
                logging.info(f"  [SUCCESS] Found matching vector in Pinecone index 'river', namespace '{AGENT_NAME_TO_VERIFY}'.")
                # Optional: Log more details from the match
                match = query_response.matches[0]
                logging.info(f"    - Vector ID: {match.id}")
                logging.info(f"    - Vector Score: {match.score}")
            else:
                logging.error(f"  [FAILURE] Did NOT find a vector for source_identifier '{source_id}' in namespace '{AGENT_NAME_TO_VERIFY}'.")
                all_verified = False

        except Exception as e:
            logging.error(f"  [ERROR] An error occurred while querying Pinecone for source_identifier '{source_id}': {e}")
            all_verified = False
            
    print("\n" + "="*40)
    if all_verified and logs_to_check:
        print("  Verification successful! All checked records are correctly linked.")
    else:
        print("  Verification FAILED. Some records are not correctly linked. See logs above.")
    print("="*40)


if __name__ == "__main__":
    verify_migration()
