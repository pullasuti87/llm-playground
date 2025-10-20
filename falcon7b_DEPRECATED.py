import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- ALUSTUS (tehdään kerran per malli) ---
model_name = "tiiuae/falcon-7b"
print("Ladataan mallia...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("Malli ladattu.")

prompt = "what's the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# --- LÄMMITTELYAJO (Ei mitata, tärkeää tarkkuuden kannalta) ---
print("Suoritetaan lämmittelyajo...")
_ = model.generate(**inputs, max_new_tokens=10) # Lyhyt generointi riittää
print("Lämmittely tehty.")


# --- VARSINAINEN MITTAUS ---
print("Generoidaan tekstiä ja mitataan aikaa...")
start_time = time.time()

# Käytetään determinististä generointia (aina sama tulos) nopeusmittausta varten
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False # Tärkeä: takaa toistettavuuden
)

end_time = time.time()

# --- TULOSTEN LASKENTA JA NÄYTTÄMINEN ---
generation_time = end_time - start_time

# Laske tuotettujen tokenien määrä suoraan tensoreista (tarkin tapa)
num_input_tokens = inputs.input_ids.shape[1]
num_output_tokens = outputs.shape[1]
num_new_tokens = num_output_tokens - num_input_tokens

tokens_per_second = num_new_tokens / generation_time if generation_time > 0 else 0

# Dekoodaa vain uudet, generoidut tokenit
generated_text = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)

print("\n--- Generoitu teksti ---")
print(generated_text)
print("\n--- Suorituskyky ---")
print(f"Generointiaika: {generation_time:.2f} sekuntia")
