# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "Manipulation of Hard to Describe Directions." The research hypothesis is that directions in the residual stream that resist manipulation via explicit feature-naming prompts can be more easily manipulated by finetuning a new token embedding to reference them.

## Papers

Total papers downloaded: 24

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Activation Engineering (ActAdd) | Turner et al. | 2023 | papers/turner2023_*.pdf | Foundational steering method |
| 2 | Contrastive Activation Addition | Rimsky et al. | 2023 | papers/rimsky2023_*.pdf | Scaled CAA, key baseline |
| 3 | Refusal Single Direction | Arditi et al. | 2024 | papers/arditi2024_*.pdf | 1D refusal subspace, 519 citations |
| 4 | Mean-Centring Steering | Jorgensen et al. | 2023 | papers/jorgensen2023_*.pdf | Improved steering vectors |
| 5 | Unreliability of Steering | Braun et al. | 2025 | papers/braun2025_*.pdf | **Steering fails for non-coherent directions** |
| 6 | Angular Steering | Vu & Nguyen | 2025 | papers/vu2025_*.pdf | Rotation-based steering |
| 7 | SAE Steering Refinement | Wang et al. | 2025 | papers/wang2025_*.pdf | SAE-based vector denoising |
| 8 | Steering Thinking Models | Venhoff et al. | 2025 | papers/venhoff2025_*.pdf | Reasoning behavior steering |
| 9 | Prompt Tuning at Scale | Lester et al. | 2021 | papers/lester2021_*.pdf | Foundational soft prompts |
| 10 | Prefix-Tuning | Li & Liang | 2021 | papers/li2021_*.pdf | Continuous prefix optimization |
| 11 | **Concept Tokens** | Sastre & Rosá | 2026 | papers/concept_tokens_2026_*.pdf | **Token embedding steering** |
| 12 | Prompt Compression | Qian et al. | 2022 | papers/qian2022_*.pdf | Contrastive conditioning |
| 13 | **Representation Tuning** | Ackerman | 2024 | papers/ackerman2024_*.pdf | **Finetuning vectors into models** |
| 14 | Representation Engineering | Zou et al. | 2023 | papers/zou2023_*.pdf | RepE framework |
| 15 | Inference-Time Intervention | Li et al. | 2023 | papers/li2023_*.pdf | Truthfulness steering, 951 citations |
| 16 | Toy Models of Superposition | Elhage et al. | 2022 | papers/elhage2022_*.pdf | Feature superposition theory |
| 17 | SAEs Find Features | Cunningham et al. | 2023 | papers/cunningham2023_*.pdf | Interpretable SAE features |
| 18 | Linear Representation Hypothesis | Park et al. | 2024 | papers/park2024_*.pdf | Formalizes linear rep hypothesis |
| 19 | Geometry of Truth | Marks & Tegmark | 2023 | papers/marks2023_*.pdf | Truth geometry in LLMs |
| 20 | From Directions to Regions | Shafran et al. | 2026 | papers/shafran2026_*.pdf | Non-linear concept structure |
| 21 | Convergent Misalignment | Soligo et al. | 2025 | papers/soligo2025_*.pdf | Convergent linear misalignment |
| 22 | Multi-property Steering | Scalena et al. | 2024 | papers/scalena2024_*.pdf | Multi-attribute control |
| 23 | Conceptor Steering | Postmus & Abreu | 2024 | papers/postmus2024_*.pdf | Ellipsoidal steering regions |
| 24 | Scaling Monosemanticity | Templeton et al. | 2024 | papers/templeton2024_*.pdf | Anthropic SAE scaling |

See papers/README.md for detailed descriptions.

## Datasets

Total datasets downloaded: 2 (+ behavioral data in code repos)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA | HuggingFace | 817 val | Truthfulness QA | datasets/truthfulqa/ | Standard steering eval benchmark |
| HotpotQA | HuggingFace | 1K val subset | Multi-hop QA | datasets/hotpotqa/ | Used in Concept Tokens paper |

Additional behavioral data available in cloned repositories:
- `code/angular_steering/data/`: True facts, truncated outputs
- `code/representation_tuning/data/`: Morally ambiguous prompts, instrumental lying prompts, paired evaluation data

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories

Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Concept Tokens | github.com/nsuruguay05/concept_tokens | Token embedding steering | code/concept_tokens/ | Core method |
| Representation Tuning | github.com/cma1114/representation_tuning | Finetuning steering vectors | code/representation_tuning/ | Dual cosine+token loss |
| Angular Steering | github.com/lone17/angular-steering | Rotation-based steering | code/angular_steering/ | Alternative approach |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy

1. Used paper-finder service with diligent mode for three topic areas:
   - "activation steering residual stream directions language models" → 102 papers
   - "soft prompt tuning learned token embeddings steering language model representations" → 63 papers
   - "linear representation hypothesis features superposition polysemantic neurons language models" → 204 papers
2. Filtered by relevance score (≥2) and direct applicability to hypothesis
3. Deep-read the most relevant papers (Concept Tokens, Representation Tuning, Unreliability of Steering)
4. Downloaded 24 papers spanning activation steering, soft prompt tuning, and mechanistic interpretability

### Selection Criteria

- Papers establishing foundational methods (ActAdd, CAA, ITI, RepE)
- Papers demonstrating limitations of current steering (Braun 2025 — unreliability)
- Papers proposing alternative steering mechanisms (Concept Tokens, Rep Tuning, Angular Steering)
- Papers on linear/non-linear representation theory (superposition, linear rep hypothesis, directions to regions)
- Code availability preferred for experimental replication

### Challenges Encountered

- The "Concept Tokens" paper (Sastre & Rosá, 2026) required arXiv API search to find the correct ID
- Some arXiv IDs in the paper-finder results mapped to different papers (e.g., 2411.02269 was a graphene paper)
- CAA behavioral datasets are not available on HuggingFace as standalone datasets; they exist within cloned repos

### Gaps and Workarounds

- **No existing dataset specifically for "hard to describe" features**: Will need to construct one using SAE features that lack clean natural language descriptions
- **No systematic benchmark for steering difficulty**: Could construct one by measuring cosine similarity consistency across prompt formulations (following Braun et al.)

## Recommendations for Experiment Design

1. **Primary dataset(s)**:
   - TruthfulQA for validating the approach on a well-studied describable behavior
   - Custom SAE-feature-based evaluation for the hard-to-describe component
   - HotpotQA for hallucination steering comparison with Concept Tokens

2. **Baseline methods**:
   - CAA (contrastive activation addition) with various prompt formulations
   - Explicit prompting ("be more X", "don't do Y")
   - In-context definition providing
   - Concept Tokens with definitional corpus
   - Unmodified model

3. **Evaluation metrics**:
   - Behavioral shift magnitude (activation projection onto target direction)
   - Task-specific accuracy (TruthfulQA score, hallucination rate)
   - Steering effectiveness vs. describability correlation
   - Perplexity preservation (WikiText)
   - Cosine similarity between learned token activations and target directions

4. **Code to adapt/reuse**:
   - `concept_tokens` repo for token embedding learning (core experimental infrastructure)
   - `representation_tuning` repo for activation hooking and direction extraction
   - `angular_steering` repo for evaluation data and alternative steering comparison
   - TransformerLens or nnsight for mechanistic analysis (not cloned — available via pip)
