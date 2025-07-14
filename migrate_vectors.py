import os
from dotenv import load_dotenv
from utils.pinecone_utils import init_pinecone

# --- Configuration ---
# Set to False to perform the actual migration.
# It is highly recommended to run with DRY_RUN = True first.
DRY_RUN = True

# Your single, shared index name.
SHARED_INDEX_NAME = "river" 

# List of agent names whose old indices need to be migrated.
# IMPORTANT: DO NOT include "river" in this list.
AGENTS_TO_MIGRATE = ["yggdrasil", "neonvoid", "newco", "ikea"] # <-- Replace with your agent names
# ---------------------

def migrate_data():
    pc = init_pinecone()
    if not pc:
        print("Pinecone client not initialized. Aborting.")
        return

    try:
        shared_index = pc.Index(SHARED_INDEX_NAME)
        print(f"Successfully connected to shared destination index: '{SHARED_INDEX_NAME}'")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to the destination index '{SHARED_INDEX_NAME}'. Please ensure it exists. Error: {e}")
        return

    for agent_name in AGENTS_TO_MIGRATE:
        try:
            old_index_name = agent_name
            target_namespace = agent_name # The agent's name becomes the namespace

            print(f"\n--- Migrating Index '{old_index_name}' -> Namespace '{target_namespace}' in Index '{SHARED_INDEX_NAME}' ---")
            
            try:
                old_index = pc.Index(old_index_name)
                stats = old_index.describe_index_stats()
            except Exception:
                print(f"Warning: Could not find or connect to old index '{old_index_name}'. Skipping.")
                continue

            if stats.total_vector_count == 0:
                print(f"Index '{old_index_name}' is empty. Nothing to migrate.")
                continue

            # Fetch all vector IDs from the old index.
            print(f"Fetching all vector IDs from namespace '{target_namespace}'...")
            vector_ids = []
            # Paginate through all vectors in the namespace
            for ids_batch in old_index.list(namespace=target_namespace, limit=99):
                vector_ids.extend(ids_batch)
            
            if not vector_ids:
                print(f"Could not list any vector IDs from namespace '{target_namespace}' in index '{old_index_name}'. Skipping.")
                continue

            print(f"Found {len(vector_ids)} vectors to migrate from '{old_index_name}'.")
            
            # Fetch and upsert in batches
            batch_size = 100
            for i in range(0, len(vector_ids), batch_size):
                batch_ids = vector_ids[i:i + batch_size]
                print(f"  Processing batch {i//batch_size + 1} of {len(vector_ids)//batch_size + 1}...")
                
                fetched_data = old_index.fetch(ids=batch_ids, namespace=target_namespace)
                vectors_to_upsert = []
                for vec_id, vec_data in fetched_data.vectors.items():
                    vectors_to_upsert.append({
                        "id": vec_id,
                        "values": vec_data.values,
                        "metadata": vec_data.metadata # This preserves the Supabase link!
                    })
                
                if vectors_to_upsert:
                    print(f"    Prepared {len(vectors_to_upsert)} vectors for upsert into namespace '{target_namespace}'.")
                    if not DRY_RUN:
                        shared_index.upsert(vectors=vectors_to_upsert, namespace=target_namespace)
                        print("    SUCCESS: Batch upserted.")
                    else:
                        print("    DRY RUN: Would have upserted this batch.")

            print(f"Finished migration for agent '{agent_name}'.")

        except Exception as e:
            import traceback
            print(f"ERROR migrating data for agent '{agent_name}': {e}")
            traceback.print_exc()

def main():
    load_dotenv()
    if DRY_RUN:
        print("="*40 + "\n    RUNNING IN DRY RUN MODE    \n" + "="*40)
        print("No data will be written. Review the output to ensure correctness.")
    else:
        print("="*40 + "\n    !!! LIVE MIGRATION MODE !!!    \n" + "="*40)
        print("This will write data to your Pinecone index. Ensure you have a backup.")
    migrate_data()
    print("\nMigration process finished.")

if __name__ == "__main__":
    main()
