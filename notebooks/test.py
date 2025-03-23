# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
def check_gpu():
    """Check and return the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    else:
        return "cpu"

check_gpu()

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llama_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", quantize=False):
    """
    Load a LLAMA model variant using PyTorch.
    
    Args:
        model_name: The model to load from Hugging Face
        quantize: Whether to apply 4-bit quantization
        
    Returns:
        model: The loaded model
        tokenizer: The tokenizer for the model
    """
    print(f"Loading model: {model_name}")
    device = check_gpu()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up quantization options if enabled
    if quantize:
        print("Loading with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        # Load without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # Explicitly move model to GPU if available
    model.to(device)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    return model, tokenizer


# %%

def generate_text(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7):
    """
    Generate text using the loaded model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer for the model
        prompt: The input prompt to generate from
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Controls randomness (lower is more deterministic)
        
    Returns:
        The generated text as a string
    """
    # Prepare the model inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# %%
# Choose model and whether to apply quantization
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
use_quantization = False  # Set to True for larger models

# %%
model, tokenizer = load_llama_model(model_name, quantize=use_quantization)


# %%

# Try a simple prompt
prompt = "Explain Go programming language in simple terms:"

print("\nPrompt:", prompt)
print("\nGenerating response...")

response = generate_text(model, tokenizer, prompt)

print("\nGenerated response:")
print(response)

# %%
print(next(model.parameters()).device)  # Should print "cuda:0"


# %%
import torch
print("CUDA Available:", torch.cuda.is_available())  # Should print True
print("CUDA Device Count:", torch.cuda.device_count())  # Should be > 0
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
print("MPS Available:", torch.backends.mps.is_available())  # For Apple M1/M2 GPUs



