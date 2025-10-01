import argparse
from datetime import datetime
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List
import time
import psutil

CHROMA_PATH = "chroma"
DATA_PATH = "data"
MEM_PATH = "memory"

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks: List[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks: List[Document]):
    """Add documents to Chroma vector store"""
    if not chunks:
        print("No chunks to process")
        return

    # Get embedding function
    embedding_function = get_embedding_function()
    
    try:
        # Check if database exists
        if os.path.exists(CHROMA_PATH):
            # Load existing database
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_function
            )
        else:
            # Create new database with initial documents
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=CHROMA_PATH
            )
            return db

        # Calculate Page IDs for new chunks
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Get existing documents
        existing_items = db.get()
        existing_ids = set(existing_items["ids"]) if existing_items else set()
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Filter new chunks
        new_chunks = [
            chunk for chunk in chunks_with_ids 
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            print("✅ Documents added successfully")
        else:
            print("✅ No new documents to add")

        return db

    except Exception as e:
        print(f"Error in add_to_chroma: {str(e)}")
        raise
    
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def add_to_memory_chroma(text: str, metadata: dict = None):
    """Add a single text entry to the Chroma vector store as memory"""
    if not text:
        print("Text is required to add to memory")
        return
    
    embedding_function = get_embedding_function()

    try:
        memory_doc = Document(page_content=text, metadata=metadata or {'type': 'memory', 'timestamp': str(datetime.now())})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        mem_chunks = text_splitter.split_documents([memory_doc])

        if os.path.exists(MEM_PATH):
            mem_db = Chroma(
                persist_directory=MEM_PATH,
                embedding_function=embedding_function
            )
        else:
            mem_db = Chroma.from_documents(
                documents=mem_chunks,
                embedding=embedding_function,
                persist_directory=MEM_PATH
            )
            return mem_db

        chunk_with_ids = calculate_chunk_ids(mem_chunks)
        mem_db.add_documents(chunk_with_ids, ids=[chunk.metadata["id"] for chunk in chunk_with_ids])
        print(f"✅ Memory added successfully: {len(chunk_with_ids)} chunks")
        return mem_db

    except Exception as e:
        print(f"Error in add_to_memory_chroma: {str(e)}")
        raise


def clear_memory():
    """Clear the memory database by safely closing connections and removing files"""
    if os.path.exists(MEM_PATH):
        try:
            embedding_function = get_embedding_function()
            mem_db = Chroma(
                persist_directory=MEM_PATH,
                embedding_function=embedding_function
            )

            collection_name = mem_db._collection.name
            mem_db._client.delete_collection(collection_name)
            
            mem_db = None
            time.sleep(2)  
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(MEM_PATH, ignore_errors=True)
                    if not os.path.exists(MEM_PATH):
                        print("🧹 Memory cleared successfully")
                        return
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait between attempts
                        continue
            
            # If normal deletion fails, try force cleanup
            print("⚠️ Normal cleanup failed, attempting force cleanup...")
            for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                try:
                    for file in proc.open_files():
                        if MEM_PATH in str(file.path):
                            proc.kill()
                            time.sleep(1)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
            
            shutil.rmtree(MEM_PATH, ignore_errors=True)
            print("🧹 Memory forcefully cleared")
            
        except Exception as e:
            print(f"❌ Error clearing memory: {str(e)}")
            print("Please restart the application and try again.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    print(f"📚 Loaded {len(documents)} documents")
    chunks = split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()