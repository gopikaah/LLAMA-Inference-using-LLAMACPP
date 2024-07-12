# model_loader.py

from accelerate import Accelerator
from langchain_community.llms import LlamaCpp

# Initialize the Accelerator
accelerator = Accelerator()

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=40,
    n_batch=512,
    verbose=False,
)

def get_model():
    return llm, accelerator
