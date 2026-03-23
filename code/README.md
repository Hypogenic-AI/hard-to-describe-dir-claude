# Cloned Repositories

## 1. Concept Tokens
- **URL**: https://github.com/nsuruguay05/concept_tokens
- **Location**: `code/concept_tokens/`
- **Purpose**: Implementation of Concept Tokens (Sastre & Rosá, 2026) — learning behavioral embeddings through concept definitions in frozen LLMs
- **Key Functionality**:
  - Adds a new special token to a pretrained LLM
  - Optimizes only its embedding using a definitional corpus
  - Supports directional use (assert/negate) at inference time
- **Requirements**: PyTorch, Transformers, likely requires GPU for training
- **Relevance**: Core method for the research hypothesis — can be adapted to learn tokens that reference hard-to-describe directions

## 2. Representation Tuning
- **URL**: https://github.com/cma1114/representation_tuning
- **Location**: `code/representation_tuning/`
- **Purpose**: Finetuning steering vectors directly into LLMs (Ackerman, 2024)
- **Key Files**:
  - `steering.ipynb`: Main steering experiments
  - `custom_fine_tune.ipynb`: Fine-tuning with dual cosine similarity + token loss
  - `enhanced_hooking.py`: Activation hooking utilities
  - `data/`: Evaluation prompts (morally ambiguous, instrumental lying)
  - `directions_*.pkl`: Pre-computed steering vectors for Llama-2-13b-chat and Llama-3-8b
- **Requirements**: PyTorch, Transformers, Llama-2-13b-chat or Llama-3-8b
- **Relevance**: Demonstrates the bridge between steering vectors and finetuning. The cosine similarity loss approach could be combined with concept token learning.

## 3. Angular Steering
- **URL**: https://github.com/lone17/angular-steering
- **Location**: `code/angular_steering/`
- **Purpose**: Angular Steering (Vu & Nguyen, 2025) — behavior modulation via rotation in activation space
- **Key Files**:
  - `data/`: Evaluation datasets (true facts, truncated outputs)
  - `output/`: Pre-computed steering results for various models
- **Requirements**: PyTorch, Transformers
- **Relevance**: Alternative geometric approach to steering that may handle hard-to-describe directions differently than additive methods
