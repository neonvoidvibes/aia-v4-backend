import os
import sys
import argparse
import pinecone
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_and_upsert(namespace):
    """
    Creates a dummy file, upserts it to a specific Pinecone namespace, and then deletes the file.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        sys.exit(1)

    # Define Pinecone index name
    index_name = "river"

    # Create a dummy file
    file_path = "hello_world.txt"
    try:
        with open(file_path, "w") as f:
            f.write("Hello World")
        print(f"'{file_path}' created.")

        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=api_key)

        # Load the document
        loader = TextLoader(file_path)
        documents = loader.load()
        print("Document loaded.")

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        print("Embeddings created.")

        # Upsert the document to the specified namespace
        PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name, namespace=namespace)
        print(f"Successfully upserted to namespace: '{namespace}'")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Delete the dummy file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"'{file_path}' deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsert a 'Hello World' txt file to a selected Pinecone namespace.")
    parser.add_argument("--namespace", required=True, help="The Pinecone namespace to upsert the document to.")
    args = parser.parse_args()

    create_and_upsert(args.namespace)
