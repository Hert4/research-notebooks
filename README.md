# Evolution Sim Code

A simplified Python implementation of **Evolutionary Model Merging** — using evolutionary algorithms to find optimal weight combinations for merging multiple large language models (LLMs).

Inspired by the paper on evolutionary optimization for model merging, this project provides a lightweight, easy-to-run implementation that works out-of-the-box on **Kaggle** and **Google Colab** free-tier environments.

## How It Works

The algorithm evolves a population of **weight vectors** (one weight per source model) across multiple generations to find the combination that produces the best merged model, measured by **perplexity** on a held-out evaluation set.

```
Generation Loop:
  1. For each weight vector in the population:
     - Merge source models using weighted average of parameters
     - Evaluate merged model's perplexity on GSM8K
  2. Select elites (top 20% lowest perplexity)
  3. Create next generation via:
     - Crossover (average of two tournament-selected parents)
     - Mutation (Gaussian noise with probability `mutation_rate`)
     - Normalize weights to sum to 1
```

### Key Design Choices

| Aspect | Implementation |
|---|---|
| **Merge strategy** | Weighted average in parameter space |
| **Fitness signal** | Perplexity on GSM8K (proxy metric) |
| **Selection** | Tournament selection (size 3) |
| **Elitism** | Top 20% carried to next generation |
| **Memory management** | Models kept on CPU; moved to GPU only during evaluation |

## Requirements

- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- tqdm

```bash
pip install torch transformers datasets numpy tqdm
```

> **Note:** On Kaggle/Colab, `transformers` and `datasets` are pre-installed.

## Quick Start

### Basic Usage

```python
python evol.py
```

This runs the default configuration which merges two `Qwen/Qwen3-0.6B` models as a demonstration.

### Custom Models

Edit the `models_spec` in `main()` or call `evolutionary_merge()` directly:

```python
from evol import evolutionary_merge

models_spec = [
    {'name': 'Qwen/Qwen3-0.6B'},
    {'name': 'Qwen/Qwen3-0.6B'},
    # Add more models (requires more GPU memory)
]

best_weights, best_score, merged_model, tokenizer = evolutionary_merge(
    models=models_spec,
    population_size=10,    # Number of candidates per generation
    generations=5,         # Number of evolutionary generations
    mutation_rate=0.2,     # Probability of mutation
    eval_samples=50        # Number of GSM8K samples for evaluation
)
```

### Using the Merged Model

```python
# After merging, generate text with the optimized model
inputs = tokenizer("Question: What is 25 * 4?\nAnswer:", return_tensors="pt").to("cuda:0")
outputs = merged_model.generate(**inputs, max_length=100)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `population_size` | `10` | Number of weight vectors per generation |
| `generations` | `5` | Number of evolutionary generations |
| `mutation_rate` | `0.2` | Probability of applying Gaussian mutation to a child |
| `eval_samples` | `50` | Number of GSM8K samples used for perplexity evaluation |

## Limitations

- **Parameter-space merging only** — data-flow and layer-routing optimizations from the original research paper are not included.
- **Same architecture required** — all source models must share the same architecture and parameter names.
- **Proxy fitness** — perplexity on GSM8K is a lightweight proxy; swap `evaluate_perplexity()` for task-specific metrics for better results.
- **Memory** — merging large models (7B+) requires significant RAM/VRAM. The free-tier GPU on Kaggle/Colab works well for smaller models (~0.6B–1.5B).

## License

Apache License 2.0 — see the [LICENSE header](evol.py) for details.

## Citation

If you use this code in your research or project, please cite:

```
Hert4 (2025). Evolutionary Model Merging (simplified implementation).
GitHub: https://github.com/Hert4/evolution-sim-code
```
