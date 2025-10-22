"""THIS NEEDS OPTIMIZING. NOT WORKING"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

t  = [
#"cerebras/Cerebras‑GPT‑111M",  
#"cerebras/Cerebras‑GPT‑256M", 
#"cerebras/Cerebras‑GPT‑590M", 
#"cerebras/Cerebras‑GPT‑1.3B", 
#"cerebras/Cerebras‑GPT‑2.7B", 
#"cerebras/Cerebras‑GPT‑6.7B", 
#"cerebras/Cerebras‑GPT‑13B",
]

MAX_TOKENS_PER_TASK = 100

tasks = [
    {
        "name": "1. bullet list",
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
        "name": "4. code",
        "prompt": "A Python function that adds two numbers looks like this:",
    },
    {
        "name": "5. reasoning",
        "prompt": "In one or two sentences, explain why using an LLM is better than a rule-based chatbot.",
    }
]

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

for i in t:
    print(f"\n{'='*50}\nPROCESSING MODEL: {i}\n{'='*50}")

    model_name = i.split("/")[-1]
    output_filepath = os.path.join(results_dir, f"{model_name}.txt")

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(f"Evaluation Results for Model: {i}\n")
            f.write(f"{'='*40}\n\n")

            print(f"Loading model: {i}...")
            tokenizer = AutoTokenizer.from_pretrained(i)
            
            # might not have pad_token set, need to use eos_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                i,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            print("Model loaded successfully.")

            print("Performing a warmup run...")
            warmup_prompt = "Hello, world!"
            inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)
            _ = model.generate(**inputs, max_new_tokens=10)
            print("Warmup complete.")

            for task in tasks:
                print(f"\nExecuting task: {task['name']}")
                f.write(f"--- Task: {task['name']} ---\n")
                f.write(f"Prompt: {task['prompt']}\n\n")

                prompt = task["prompt"]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                start_time = time.time()

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS_PER_TASK,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                end_time = time.time()

                generation_time = end_time - start_time
                num_input_tokens = inputs.input_ids.shape[1]
                num_output_tokens = outputs.shape[1]
                num_new_tokens = num_output_tokens - num_input_tokens

                tokens_per_second = 0
                if num_new_tokens > 0 and generation_time > 0:
                    tokens_per_second = num_new_tokens / generation_time

                generated_text = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)

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
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(error_message)
        except:
            pass

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