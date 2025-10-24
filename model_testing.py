import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

t = [
    #"cerebras/Cerebras-GPT-111M",
    #"cerebras/Cerebras-GPT-256M",
    #"cerebras/Cerebras-GPT-590M",
    #"cerebras/Cerebras-GPT-1.3B",
    #"cerebras/Cerebras-GPT-2.7B",
    #"cerebras/Cerebras-GPT-6.7B",
    #"cerebras/Cerebras-GPT-13B",
    #"databricks/dolly-v2-12b",
    #"tiiuae/falcon-7b",
    #"MathLLMs/MathCoder-L-7B",
    #"MathLLMs/MathCoder-L-13B",
    #"mosaicml/mpt-7b",
    #"OrionStarAI/Orion-14B-Base",
    #"FreedomIntelligence/phoenix-chat-7b",
    #"DAMO-NLP-MT/polylm-13b",
    #"EleutherAI/pythia-70m",
    #"EleutherAI/pythia-160m",
    #"EleutherAI/pythia-410m",
    #"EleutherAI/pythia-1b",
    #"EleutherAI/pythia-1.4b",
    #"EleutherAI/pythia-2.8b",
    #"EleutherAI/pythia-6.9b",
    #"EleutherAI/pythia-12b",
    #"openbmb/UltraLM-13b",
    #"vanillaOVO/WizardLM-7B-V1.0",
    #"CohereLabs/aya-101",
    #"BioMistral/BioMistral-7B", # huom task 1 tehty
    #"daven3/k2",
    #"m-a-p/neo_2b_general",
    #"m-a-p/neo_7b",
    #"Xianjun/PLLaMa-7b-base",
    #"OrionZheng/openmoe-8b",
    #"TinyLlama/TinyLlama_v1.1",
    #"OrionZheng/openmoe-34b-200B",
    #"tiiuae/falcon-40b",

    # HUOM, tee biomistrak
]

max_tokens = 150

tasks = [
    {
        "name": "1. bulleted List",
        "prompt": "List three common computer hardware components as a bulleted list:",
    },
    {
        "name": "2. sentence completion",
        "prompt": "The primary function of a computer keyboard is to",
    },
    {
        "name": "3. summarization",
        "prompt": """Summarize the following text in a single sentence:

Python is a high-level, interpreted programming language known for its easy readability and clean design. Its syntax closely resembles plain English, making it simple for beginners to learn and easy for developers to maintain. This combination of simplicity and readability makes Python a general-purpose language with widespread applications in fields like web development, data science, artificial intelligence, and automation.""",
    },
    {
        "name": "4. code generation",
        "prompt": "A Python function that adds two numbers looks like this:",
    },
    {
        "name": "5. reasoning",
        "prompt": "In one or two sentences, explain why using an LLM is better than a rule-based chatbot.",
    }
]

results_f = "results"
os.makedirs(results_f, exist_ok=True)


for i in t:
    print(f"\n{'='*50}\nPROCESSING MODEL: {i}\n{'='*50}")

    safe_i = i.replace("/", "_")
    output_filepath = os.path.join(results_f, f"{safe_i}.txt")

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"Evaluation Results for Model: {i}\n")
            f.write(f"{'='*40}\n\n")

            print(f"Loading model: {i}...")

            if "MathCoder" in i:
                print("MathCoder model detected. Loading with use_fast=False.")
                tokenizer = AutoTokenizer.from_pretrained(i, use_fast=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(i, legacy=False)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if "aya-101" in i:
                print("T5-based model detected. Loading with AutoModelForSeq2SeqLM.")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    i,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                )
            else:
                print("CausalLM model detected. Loading with AutoModelForCausalLM.")
                model = AutoModelForCausalLM.from_pretrained(
                    i,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    #trust_remote_code=True,
                    load_in_4bit=True,
                )

            print("Model loaded successfully.")

            # --- WARMUP RUN (Important for accurate performance measurement) ---
            print("Performing a warmup run...")
            warmup_prompt = "Hello, world!"
            inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)
            _ = model.generate(**inputs, max_new_tokens=10)
            print("Warmup complete.")

            # --- EXECUTE AND MEASURE EACH TASK ---
            for task in tasks:
                print(f"\nExecuting task: {task['name']}")
                f.write(f"--- Task: {task['name']} ---\n")
                f.write(f"Prompt: {task['prompt']}\n\n")

                prompt = task["prompt"]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # --- PERFORMANCE MEASUREMENT ---
                start_time = time.time()

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens
                ,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                end_time = time.time()

                # --- CALCULATE AND RECORD RESULTS ---
                generation_time = end_time - start_time
                num_input_tokens = inputs.input_ids.shape[1]
                num_output_tokens = outputs.shape[1]

                # For CausalLM, the output includes the prompt, so we subtract it.
                # For Seq2SeqLM, the output is only the generated sequence.
                if model.config.is_encoder_decoder:
                    num_new_tokens = num_output_tokens
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    num_new_tokens = num_output_tokens - num_input_tokens
                    generated_text = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)


                tokens_per_second = 0
                if num_new_tokens > 0 and generation_time > 0:
                    tokens_per_second = num_new_tokens / generation_time

                # Write results to the file
                f.write("--- Model's Response ---\n")
                f.write(generated_text + "\n\n")
                f.write("--- Performance ---\n")
                f.write(f"Generation time: {generation_time:.2f} seconds\n")
                f.write(f"Generated new tokens: {num_new_tokens}\n")
                f.write(f"Speed: {tokens_per_second:.2f} tokens/second\n")
                f.write(f"{'-'*40}\n\n")

            print(f"\nAll tasks for {i} completed.")

    except Exception as e:
        error_message = f"An error occurred while processing {i}: {e}"
        print(error_message)
        # Write the error to the file if it was opened
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(error_message)
        except:
            pass # If file opening itself failed, there's nothing to do

    finally:
        print(f"Cleaning up memory for model {i}...")
        try:
            del model
            del tokenizer
            torch.cuda.empty_cache()
            print("Cleanup successful.")
        except NameError:
            print("Model was not fully loaded, skipping cleanup.")

    print(f"Results for {i} have been saved to {output_filepath}")

print(f"\n{'='*50}\nAll models have been processed.\n{'='*50}")