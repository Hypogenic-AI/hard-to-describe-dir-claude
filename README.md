# Manipulation of Hard-to-Describe Directions

Can learned token embeddings manipulate residual stream directions that resist explicit prompting?

## Key Findings

- **Concept tokens work for all tested features**: Learned embeddings successfully shift residual stream activations toward target directions for all 6 behavioral features (sentiment, formality, verbosity, sycophancy, AI-sounding, hedging)
- **Hard features benefit most**: For hard-to-describe features (AI-sounding, hedging, sycophancy), concept tokens outperform explicit prompting by 25-106%
- **Easy features are well-served by prompting**: For easily describable features (formality, verbosity), explicit prompting outperforms concept tokens
- **Large effect size**: Spearman rho=0.71 between feature difficulty and concept token advantage over prompting
- **Supports the hypothesis**: The advantage of learned token embeddings increases as features become harder to describe in natural language

## Methodology

1. **Direction extraction**: Compute feature directions from contrastive text pairs in Pythia-410M's residual stream (layer 12)
2. **Describability measurement**: Quantify how consistently multiple prompt variants activate each direction
3. **Concept token training**: Optimize a single embedding vector (added to BOS position) to shift activations toward the target direction
4. **Comparison**: Evaluate concept tokens vs. explicit prompting vs. CAA across 20 neutral test texts

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformers transformer-lens numpy scipy matplotlib seaborn pandas tqdm einops jaxtyping

# Run experiment (~5 min on RTX A6000)
CUDA_VISIBLE_DEVICES=0 python src/experiment.py
```

## File Structure

```
├── REPORT.md              # Full research report with analysis
├── README.md              # This file
├── planning.md            # Research plan
├── literature_review.md   # Literature survey
├── resources.md           # Resource catalog
├── src/
│   └── experiment.py      # Main experiment code
├── results/
│   ├── experiment_results.json
│   ├── describability_scores.json
│   ├── statistical_tests.json
│   ├── config.json
│   └── plots/
│       ├── main_results.png
│       ├── per_feature_distributions.png
│       └── method_scatter.png
├── papers/                # Downloaded research papers
├── datasets/              # TruthfulQA, HotpotQA
└── code/                  # Cloned baseline repositories
```

See [REPORT.md](REPORT.md) for full experimental details and analysis.
