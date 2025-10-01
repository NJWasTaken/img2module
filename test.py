import process_docs as pd
import query_docs as qd

while True:
    user_input = input("Enter your query (or type 'exit' to quit, 'clear' to clear memory): ")
    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'clear':
        pd.clear_memory()
        continue
    response = qd.query_rag(user_input)
    print(f"Response: {response}\n")
    mem = "User input: " + user_input + "\nResponse: " + response
    pd.add_to_memory_chroma(mem)