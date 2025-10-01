import argparse
import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from process_docs import MEM_PATH, CHROMA_PATH

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
This is your memory of the conversation so far. Use it to provide better answers.

{memory}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def get_relevant_memories(query: str, k: int = 3) -> str:
    """Retrieve relevant memories for a given query"""
    if not os.path.exists(MEM_PATH):
        return ""

    try:
        # Get embedding function
        embedding_function = get_embedding_function()
        
        # Load memory database
        memory_db = Chroma(
            persist_directory=MEM_PATH,
            embedding_function=embedding_function
        )

        # Search for relevant memories
        results = memory_db.similarity_search_with_score(query, k=k)
        
        # Format memories with timestamps
        memories = []
        for doc, score in results:
            timestamp = doc.metadata.get('timestamp', 'Unknown time')
            memories.append(f"[{timestamp}] {doc.page_content}")

        return "\n\n".join(memories)

    except Exception as e:
        print(f"Error retrieving memories: {str(e)}")
        return ""

def query_rag(query_text: str) -> str:
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    memory = get_relevant_memories(query_text, k=3)
    results = db.similarity_search_with_score(query_text+'\n\n---\n\n'+memory, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    prompt = prompt_template.format(context=context_text, memory=memory, question=query_text)
    print(prompt)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()