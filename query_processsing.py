import time
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from model_loader import get_model
import sys

def handle_exception(exc_type, exc_value, exc_traceback):
    # Log the exception or handle it as needed
    print(f"Ignored exception: {exc_type}, {exc_value}")

# Set excepthook to handle all unhandled exceptions
sys.excepthook = handle_exception

# Retrieve the model and accelerator
llm, accelerator = get_model()

# Initial context
initial_context = 'You are a helpful assistant. Answer the questions asked. If you dont know the answer do not make up anything. Be precise and include factual information and evidence reagrding you'
chat_history = []

# Define the prompt template with a placeholder for the question
template = """
{history}
Context: {context}
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question", "history"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

if __name__ == "__main__":
    print('PLEASE ENTER THE QUESTION, Enter any number to stop')
    while True:
        
        question = input('Q: ')
        if question.isdigit():
            break
        start = time.time()
        try:
            # Prepare the chat history
            history = "\n".join(chat_history)
            with accelerator.autocast():
                answer = llm_chain.run({"context": initial_context, "question": question, "history": history})
            end = time.time()
            
            # Print the answer and the time taken
            print('A:',answer, '\n')
            # print(end - start, 'Seconds')
            
            # Update the chat history
            chat_history.append(f"Q: {question}")
            chat_history.append(f"A: {answer.strip()}")
            
        except Exception as e:
            print(f"An error occurred: {e}")  # Print the specific error message
            print('')  # Optionally handle or log the error further
