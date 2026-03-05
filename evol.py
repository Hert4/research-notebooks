# Copyright 2025 Hert4
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------
# If you use this code in your research or project, please cite:
#
#   Hert4 (2025). Evolutionary Model Merging (simplified implementation).
#   GitHub: https://github.com/Hert4/evolution-sim-code
#
# ----------------------------------------------------------------------------
#
# Notes:
#   - Runs on Kaggle/Colab without extra installs (transformers, datasets pre-installed)
#   - Perplexity on GSM8K is a lightweight proxy fitness signal;
#     swap evaluate_perplexity() for task-specific metrics as needed
#   - Models are kept on CPU between evaluations to fit within free-tier GPU RAM
#   - Only parameter-space merging is implemented (weighted average);
#     data-flow / layer-routing optimization from the original paper is not included

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evolutionary_merge(models, population_size=10, generations=5, mutation_rate=0.2, eval_samples=50):
    logger.info("Starting Evolutionary Model Merge (perplexity evaluation)...\n")

    # Load evaluation data (using GSM8K dataset as an example)
    gsm8k = load_dataset("openai/gsm8k", "main")
    eval_data = gsm8k["train"].select(range(eval_samples))

    # Load models (keep on CPU to save GPU memory during initialization) this important using in colab/kaggle
    loaded_models = []
    for model_spec in models:
        model = AutoModelForCausalLM.from_pretrained(
            model_spec['name'],
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        loaded_models.append(model)

    tokenizer = AutoTokenizer.from_pretrained(models[0]['name'])

    population = [np.random.dirichlet(np.ones(len(models))) for _ in range(population_size)]

    def evaluate_perplexity(candidate_model):
        """Calculate perplexity of the candidate model on eval_data"""
        candidate_model.to("cuda:0")  
        candidate_model.eval()

        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for row in eval_data:
                prompt = f"Question: {row['question']}\nAnswer:"
                gold_answer = row["answer"]

                # Full sequence = prompt + gold answer
                full_text = prompt + gold_answer
                inputs = tokenizer(full_text, return_tensors="pt").to("cuda:0")

                labels = inputs["input_ids"].clone()
                n_prompt_tokens = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                labels[:, :n_prompt_tokens] = -100  

                outputs = candidate_model(**inputs, labels=labels)
                loss = outputs.loss.item()

                n_tokens = (labels != -100).sum().item()
                total_loss += loss * n_tokens
                total_tokens += n_tokens

        candidate_model.to("cpu")
        torch.cuda.empty_cache()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return ppl

    best_weights, best_score = None, float("inf") 

    for gen in tqdm(range(generations), desc="Generations", leave=True):
        logger.info(f"Generation {gen+1}/{generations}")
        scores = []

        for weights in population:
            # Create model by merging according to weights
            candidate_model = AutoModelForCausalLM.from_pretrained(
                models[0]['name'],
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            
            with torch.no_grad():
                for name, param in candidate_model.named_parameters():
                    merged_param = torch.zeros_like(param.data, device="cpu")
                    for i, model in enumerate(loaded_models):
                        if name in dict(model.named_parameters()):
                            merged_param += weights[i] * dict(model.named_parameters())[name].data.cpu()
                    param.data = merged_param

            # Evaluate perplexity
            score = evaluate_perplexity(candidate_model)
            scores.append(score)

            # Clean up
            del candidate_model
            torch.cuda.empty_cache()

        best_idx = np.argmin(scores) 
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_weights = population[best_idx]

        logger.info(f"Generation {gen+1} best PPL: {scores[best_idx]:.5f}")

        # Generate next generation
        new_population = []
        elite_count = max(1, int(population_size * 0.2))
        elite_indices = np.argsort(scores)[:elite_count]  # Lowest perplexities
        for idx in elite_indices:
            new_population.append(population[idx])

        while len(new_population) < population_size:
            parent1 = population[np.argmin([scores[i] for i in np.random.choice(population_size, 3)])]
            parent2 = population[np.argmin([scores[i] for i in np.random.choice(population_size, 3)])]
            
            child = (parent1 + parent2) / 2
            child = child / np.sum(child)  

            if np.random.random() < mutation_rate:
                mutation = np.random.normal(0, 0.1, len(models))
                child = np.abs(child + mutation)
                child = child / np.sum(child)

            new_population.append(child)

        population = new_population

    logger.info(f"Best weights: {best_weights}, Best PPL: {best_score:.2f}")
    
    # Create final merged model with best weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    merged_model = AutoModelForCausalLM.from_pretrained(
        models[0]['name'],
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    with torch.no_grad():
        for name, param in tqdm(merged_model.named_parameters(), desc="Final merge", leave=True):
            merged_param = torch.zeros_like(param.data, device=device)
            for i, model in enumerate(loaded_models):
                if name in dict(model.named_parameters()):
                    merged_param += best_weights[i] * dict(model.named_parameters())[name].data.to(device)
            param.data = merged_param
    
    for model in loaded_models:
        del model
    torch.cuda.empty_cache()
    
    return best_weights, best_score, merged_model, tokenizer

def main():
    """
    Main function demonstrating how to use the evolutionary merge algorithm
    """
    models_spec = [
        {'name': 'Qwen/Qwen3-0.6B'}, 
        {'name': 'Qwen/Qwen3-0.6B'}   
       # added more require more memory
    ]
    
    print("Starting Evolutionary Model Merge Process...")
    print(f"Models to merge: {[model['name'] for model in models_spec]}")
    print(f"Population size: 5, Generations: 3, Eval samples: 10")
    
    try:
      
        best_weights, best_score, merged_model, tokenizer = evolutionary_merge(
            models=models_spec,
            population_size=5,
            generations=3,
            eval_samples=10 
        )
        
        print(f"\nOptimization Complete!")
        print(f"Best weights: {best_weights}")
        print(f"Best perplexity: {best_score:.4f}")
        
        # Show an example of how you might use the merged model
        print(f"\nMerged model is ready for use.")
        print("Example usage after merging:")
        print("# Generate text with merged model")
        print("# inputs = tokenizer(\"Your prompt here\", return_tensors=\"pt\").to(\"cuda:0\")")
        print("# outputs = merged_model.generate(**inputs, max_length=100)")
        print("# text = tokenizer.decode(outputs[0], skip_special_tokens=True)")
        
        return best_weights, best_score, merged_model, tokenizer
        
    except Exception as e:
        print(f"Error during evolutionary merge: {str(e)}")
        print("\nNote: This example requires compatible models with the same architecture.")
        print("Make sure to use models that have the same base architecture for merging.")
        return None, None, None, None


if __name__ == "__main__":
    best_weights, best_score, merged_model, tokenizer = main()
