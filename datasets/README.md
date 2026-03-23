# Downloaded Datasets

This directory contains datasets for the research project on manipulation of hard-to-describe directions in language models. Data files are NOT committed to git due to size. Follow download instructions below.

## Dataset 1: TruthfulQA (Generation)

### Overview
- **Source**: HuggingFace `truthfulqa/truthful_qa` (generation config)
- **Size**: 817 validation samples
- **Format**: HuggingFace Dataset
- **Task**: Open-ended question answering to evaluate truthfulness
- **Splits**: validation (817)
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "generation")
dataset.save_to_disk("datasets/truthfulqa/generation")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthfulqa/generation")
```

### Notes
- Standard benchmark for evaluating truthfulness in LLMs
- Used in Inference-Time Intervention (Li et al., 2023) and Representation Tuning (Ackerman, 2024)
- Contains questions designed to elicit common misconceptions
- Fields: type, category, question, best_answer, correct_answers, incorrect_answers, source

---

## Dataset 2: HotpotQA (Validation Subset)

### Overview
- **Source**: HuggingFace `hotpotqa/hotpot_qa` (distractor config)
- **Size**: 1,000 samples (from 7,405 validation total)
- **Format**: HuggingFace Dataset
- **Task**: Multi-hop question answering
- **License**: CC BY-SA 4.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation[:1000]")
dataset.save_to_disk("datasets/hotpotqa/validation_1k")
```

### Notes
- Used in Concept Tokens (Sastre & Rosá, 2026) for evaluating hallucination steering
- Multi-hop questions derived from Wikipedia articles
- Provides question, gold answer, and supporting context
- Useful for closed-book QA evaluation (discard context for steering experiments)

---

## Usage for Experiments

For this research on hard-to-describe directions:

1. **TruthfulQA**: Use as evaluation benchmark for steering effectiveness. Generate contrastive pairs (true/false claims) for extracting steering vectors. Standard metric for truthfulness steering.

2. **HotpotQA**: Use for evaluating concept token steering of hallucinations. Compare concept-token-based steering vs. prompt-based steering vs. activation steering.

3. **CAA Behavioral Data**: Available in `code/angular_steering/data/` and `code/representation_tuning/data/` - contrastive pairs and evaluation prompts for activation steering experiments.
