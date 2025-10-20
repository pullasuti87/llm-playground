import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

m = "TinyLlama/TinyLlama_v1.1"
print("ladataan malli...")

tokenizer = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(
    m,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("malli ladattu.")

prompt = "what's the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# --- lämmittely
print("lämmittely...")
_ = model.generate(**inputs, max_new_tokens=10)
print("lämmittely tehty")


# --- varsinainen ajo ---
print("generoidaan tekstiä...")
start_time = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=100,  # tämä riittää
    do_sample=False
)

end_time = time.time()

generation_time = end_time - start_time

num_input_tokens = inputs.input_ids.shape[1]
num_output_tokens = outputs.shape[1]
num_new_tokens = num_output_tokens - num_input_tokens

tokens_per_second = num_new_tokens / generation_time if generation_time > 0 else 0

generated_text = tokenizer.decode(
    outputs[0][num_input_tokens:], skip_special_tokens=True)

print("\n--- generoitu teksti ---")
print(generated_text)
print("\n--- AIKA ---")
print(f"generointiaika: {generation_time:.2f} sekuntia")
