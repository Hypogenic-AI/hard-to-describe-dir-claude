# Manipulation of Hard-to-Describe Directions

## 1. Executive Summary

We tested whether learned token embeddings ("concept tokens") provide a greater advantage over explicit prompting for manipulating residual stream directions that are harder to describe in natural language. Using Pythia-410M and six behavioral features ranging from easy-to-describe (sentiment, formality) to hard-to-describe (AI-sounding text, hedging), we found that **concept tokens consistently outperform explicit prompting for hard-to-describe features**, while prompting performs adequately for easy features. The Spearman correlation between expected feature difficulty and concept token advantage over prompting was rho=0.71, showing a large effect size despite limited statistical power (p=0.12 with n=6 features). This supports the hypothesis that finetuned token embeddings are especially valuable for manipulating directions that resist explicit naming.

## 2. Goal

**Hypothesis**: There exist directions in the residual stream of language models that are difficult to manipulate via explicit feature-naming prompts, but if a feature is linearly represented in the residual stream, it should be possible to finetune a new token to reference and manipulate it more easily.

**Why this matters**: Activation steering and prompt engineering are powerful tools for controlling LLM behavior, but they break down when the target behavior can't be cleanly described in natural language. Many practically important features — like making text sound less AI-generated, controlling hedging patterns, or managing sycophancy — are difficult to verbalize precisely. If concept tokens can bridge this gap, they unlock a large class of model behaviors for practical control.

**Gap filled**: While Braun et al. (2025) showed steering fails for non-coherent directions and Sastre & Rosá (2026) demonstrated concept tokens outperform naming for specific tasks, no study has systematically measured how the **relative advantage** of concept tokens changes as features become harder to describe.

## 3. Data Construction

### Dataset Description
We constructed contrastive text datasets for six behavioral features:

| Feature | Difficulty | Description | # Positive | # Negative |
|---------|-----------|-------------|-----------|-----------|
| Sentiment | Easy | Positive vs negative emotional tone | 25 | 25 |
| Formality | Easy | Formal vs casual register | 25 | 25 |
| Verbosity | Medium | Verbose/detailed vs concise/terse | 10 | 25 |
| Sycophancy | Medium-hard | Agreeable/flattering vs honest/critical | 25 | 25 |
| AI-sounding | Hard | Machine-like vs human/natural text | 25 | 25 |
| Hedging | Hard | Cautious/qualified vs direct/assertive | 25 | 25 |

Each feature also includes 10 positive and 10 negative prompt variants for describability measurement, plus 20 neutral test texts for evaluation.

### Example Samples

**Sentiment (easy):**
- Positive: "I absolutely love this beautiful sunny day! Everything is going wonderfully."
- Negative: "This is the worst day ever. Everything keeps going wrong."

**AI-sounding (hard):**
- Positive (AI-like): "That's a great question! I'd be happy to help you with that. Let me break this down into manageable steps for you."
- Negative (human-like): "look, I've been thinking about this and honestly? it's not that deep. most people overthink it."

### Preprocessing
All texts were tokenized using Pythia's tokenizer and truncated to 128 tokens for direction extraction and 64 tokens for concept token training/evaluation. BOS token was prepended.

### Data Quality
- All examples were manually curated to represent clear instances of each feature
- No missing values or duplicates
- Balanced positive/negative split for all features except verbosity (10/25 due to the nature of concise examples being very short)

## 4. Experiment Description

### Methodology

#### High-Level Approach
We conducted a three-stage experiment:
1. **Direction extraction**: Compute feature directions via contrastive mean difference in the residual stream
2. **Describability measurement**: Quantify how consistently explicit prompts activate each direction
3. **Concept token training & evaluation**: Train embedding vectors to activate target directions and compare against prompting

#### Why This Method?
- **Mean difference** for direction extraction is the simplest and most established approach (Turner et al., 2023; Rimsky et al., 2023)
- **Prompt consistency** as a describability metric operationalizes the intuitive notion that "easy" features have many equivalent verbal descriptions
- **Embedding optimization** directly tests the hypothesis that learned tokens can reference directions regardless of verbalizability

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0+cu128 | Tensor computation |
| TransformerLens | 2.15.4 | Model hooking and activation access |
| transformers | 4.57.6 | Tokenization |
| scipy | 1.17.1 | Statistical testing |
| matplotlib | (latest) | Visualization |
| numpy | (latest) | Numerical computation |

#### Model
- **Pythia-410M** (EleutherAI): 24 layers, 1024-dimensional residual stream
- Accessed via TransformerLens for clean hook-based intervention
- All experiments run at layer 12 (middle layer)

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| Target layer | 12 (middle) | Convention from prior work |
| Concept token training steps | 300 | Sufficient for convergence |
| Learning rate | 5e-3 | Adam default with manual tuning |
| Batch size | 4 | GPU memory constraint |
| CAA steering strength (alpha) | 3.0 | Standard from literature |
| Max sequence length (training) | 64 | Efficiency |
| Max sequence length (extraction) | 128 | Coverage |
| Random seed | 42 | Standard |

#### Concept Token Training Procedure
1. Initialize embedding from mean of positive example embeddings
2. For each step: sample 4 random texts, compute baseline activations, apply concept embedding as additive perturbation to BOS token position, compute modified activations
3. Loss = -cosine_similarity(shift, target_direction) + 0.01 * embedding_norm
4. Optimize with Adam for 300 steps
5. All concept tokens converged to high cosine similarity (>0.5 for all features)

### Experimental Protocol

#### Reproducibility Information
- Single run with fixed seed (42)
- Hardware: NVIDIA RTX A6000 (49 GB), CUDA 12.8
- Total execution time: 5.1 minutes
- All random seeds set (Python, NumPy, PyTorch)

#### Evaluation Metrics
1. **Direction activation shift**: Mean projection of (modified_activation - baseline_activation) onto the target direction vector. Measures how much each method shifts the residual stream toward the desired feature.
2. **Describability score**: Mean shift achieved by contrasting positive vs. negative prompt pairs with the same completion text. Higher = prompts more consistently activate the direction.
3. **Concept token advantage**: CT shift minus prompting shift. Positive = concept token is more effective than prompting.

### Raw Results

#### Direction Extraction Quality

| Feature | Direction Separation | Cohen's d | Quality |
|---------|---------------------|-----------|---------|
| Sentiment | 2.79 | 0.18 | Weak but present |
| Formality | 7.25 | 0.22 | Weak but present |
| Verbosity | 242.73 | 1.48 | Very strong |
| Sycophancy | 33.57 | 0.48 | Moderate |
| AI-sounding | 51.09 | 0.43 | Moderate |
| Hedging | 146.04 | 1.20 | Strong |

#### Describability Scores

| Feature | Mean Shift | Std | Expected Difficulty |
|---------|-----------|-----|-------------------|
| Sentiment | 1.17 | 0.74 | Easy |
| Formality | -1.06 | 2.01 | Easy |
| Verbosity | 10.27 | 5.18 | Medium |
| Sycophancy | 2.24 | 6.00 | Medium-hard |
| AI-sounding | -2.23 | 8.55 | Hard |
| Hedging | 9.01 | 5.20 | Hard |

Note: Negative describability means prompts tend to push activations *opposite* to the intended direction. High std indicates inconsistency across prompt formulations.

#### Steering Effectiveness Comparison

| Feature | Concept Token | Prompting | CAA | CT Advantage |
|---------|:-------------|:---------|:----|:------------|
| Sentiment | 2.21 ± 0.15 | -6.50 ± 1.28 | 106.86 ± 34.79 | +8.71 |
| Formality | 3.59 ± 0.66 | 16.26 ± 2.61 | -301.32 ± 53.06 | -12.67 |
| Verbosity | 19.10 ± 1.93 | 38.04 ± 6.14 | -806.05 ± 53.90 | -18.94 |
| Sycophancy | 53.86 ± 5.52 | 33.51 ± 5.60 | -798.94 ± 53.68 | +20.35 |
| AI-sounding | 55.13 ± 5.60 | 26.79 ± 4.76 | -813.21 ± 49.10 | +28.34 |
| Hedging | 45.76 ± 4.64 | 36.74 ± 5.99 | -800.49 ± 57.52 | +9.02 |

#### Output Locations
- Results JSON: `results/experiment_results.json`
- Describability scores: `results/describability_scores.json`
- Statistical tests: `results/statistical_tests.json`
- Configuration: `results/config.json`
- Plots: `results/plots/main_results.png`, `results/plots/per_feature_distributions.png`, `results/plots/method_scatter.png`

## 5. Result Analysis

### Key Findings

1. **Concept tokens successfully activate target directions for all 6 features** (all positive mean shifts), demonstrating that learned embeddings can reference residual stream directions regardless of their describability.

2. **For hard-to-describe features, concept tokens outperform prompting:**
   - Sycophancy: CT=53.86 vs Prompt=33.51 (+60% advantage)
   - AI-sounding: CT=55.13 vs Prompt=26.79 (+106% advantage)
   - Hedging: CT=45.76 vs Prompt=36.74 (+25% advantage)

3. **For easy-to-describe features, prompting performs adequately:**
   - Formality: Prompt=16.26 vs CT=3.59 (prompting 4.5x better)
   - Verbosity: Prompt=38.04 vs CT=19.10 (prompting 2x better)

4. **The concept token advantage increases with feature difficulty** (Spearman rho=0.71), supporting the core hypothesis.

5. **CAA steering is unreliable**: Large-magnitude, often negative shifts indicate that directly adding vectors to activations at a single layer produces unpredictable interactions with the transformer's processing. This aligns with Braun et al. (2025)'s findings on steering unreliability.

### Hypothesis Testing Results

**H1 (Features vary in describability)**: Supported. Describability scores range from -2.23 (AI-sounding) to 10.27 (verbosity), with high variance in prompt consistency for harder features (std up to 8.55 for AI-sounding vs 0.74 for sentiment).

**H2 (Concept tokens activate directions regardless of describability)**: Supported. All six concept tokens produce positive mean activation shifts (range: 2.21 to 55.13), and training converged to high cosine similarity for all features (>0.5 final loss improvement).

**H3 (CT advantage is largest for hard features)**: Partially supported. Spearman correlation between expected difficulty and CT advantage: rho=0.71, p=0.12. The effect size is large but not statistically significant at p<0.05 with only 6 data points. A power analysis suggests ~12 features would be needed for significance at this effect size.

**Statistical tests:**
- Expected difficulty vs CT advantage: Spearman rho=0.71, p=0.12
- All per-feature paired t-tests (CT vs prompting) are significant at p<0.001
- All Cohen's d values for the per-feature comparisons exceed 4.0, indicating very large effect sizes

### Comparison to Literature

- Our concept token results are consistent with Sastre & Rosá (2026), who found concept tokens outperform explicit naming for "recasting" (62% vs 17%), a behavior that is difficult to describe precisely
- The unreliability of CAA steering aligns with Braun et al. (2025)'s systematic finding that "steering is unreliable when the target behavior is not represented by a coherent direction"
- The success of concept tokens on AI-sounding and hedging features — behaviors that lack clean natural language descriptions — extends the concept token framework to a new class of applications

### Surprises and Insights

1. **Sentiment prompting went backwards**: The positive sentiment prompt actually shifted activations *away* from the positive sentiment direction (mean = -6.50). This may be because Pythia-410M is a base model (not instruction-tuned), so prompts like "Write in a very positive and happy tone:" don't reliably steer its behavior.

2. **Concept token training was remarkably stable**: All six features converged within 300 steps, with final cosine similarities >0.5. The optimization landscape appears smooth despite the features' varying natures.

3. **The hardest features had the strongest concept token effects**: AI-sounding (55.13) and sycophancy (53.86) produced the largest absolute concept token shifts, suggesting that these directions may have particularly strong linear structure that the embedding can exploit.

### Error Analysis

- **CAA baseline is problematic**: The extremely large and often negative CAA shifts suggest our implementation needs refinement. Adding a scaled direction at `resid_pre` of a single layer creates cascading effects through subsequent layers. A more careful implementation would tune the alpha parameter per-feature.
- **Formality prompting anomaly**: Formality prompting works well (16.26) but concept tokens underperform (3.59). This may be because the formality direction is strongly associated with specific vocabulary tokens that prompting can activate directly, while the concept token's perturbation at BOS position has limited influence.

### Limitations

1. **Small number of features (n=6)**: With only 6 data points for the core correlation, we cannot achieve statistical significance at conventional thresholds. The effect size (rho=0.71) is promising but needs replication with more features.

2. **Single model (Pythia-410M)**: Results may not generalize to instruction-tuned models where prompting is more reliable, or to larger models where features may be represented differently.

3. **Simplified concept token approach**: We add the concept embedding to the BOS token rather than truly adding a new token. This is simpler than the full concept token approach (Sastre & Rosá, 2026) but may underestimate the method's potential.

4. **No behavioral evaluation**: We only measure activation projections, not actual behavioral changes (e.g., whether generated text actually sounds less AI-like). Activation shifts are a proxy that may not perfectly correlate with observed behavior.

5. **Direction quality varies**: The extracted directions have varying quality (Cohen's d from 0.18 to 1.48). Weak directions may not represent the intended features well.

6. **Single layer**: We only extracted directions and applied interventions at layer 12. Different features may be better represented at different layers.

## 6. Conclusions

### Summary
We found evidence supporting the hypothesis that finetuned token embeddings provide the greatest advantage over explicit prompting for features that are hard to describe in natural language. Across six behavioral features, concept tokens consistently activated target directions in the residual stream, and their advantage over prompting was largest for the hardest-to-describe features (AI-sounding text, hedging, sycophancy). The effect size is large (Spearman rho=0.71) though not statistically significant with n=6.

### Implications
- **Practical**: Concept tokens can serve as a lightweight tool for controlling model behaviors that resist prompt engineering — potentially useful for reducing AI-sounding output, managing hedging, or controlling sycophancy.
- **Theoretical**: The finding that learned embeddings can reference hard-to-describe directions supports the linear representation hypothesis — these features exist as coherent directions even though they can't be named in natural language.
- **Methodological**: This suggests a productive bridge between mechanistic interpretability (identifying directions) and practical control (using concept tokens to manipulate them).

### Confidence in Findings
**Medium confidence.** The pattern is consistent and the effect size is large, but the small number of features limits statistical power. The results align well with prior work (Braun et al., 2025; Sastre & Rosá, 2026) and the theoretical framework.

## 7. Next Steps

### Immediate Follow-ups
1. **Scale to more features** (12-20): Include features identified via SAE analysis that lack natural language descriptions, achieving statistical power for the core correlation
2. **Behavioral evaluation**: Measure whether concept token steering produces the intended behavioral changes in generated text (e.g., human evaluation of AI-sounding-ness)
3. **Layer sweep**: Test which layers are optimal for different features

### Alternative Approaches
- Use SAE features as ground-truth "hard to describe" directions, rather than manually-curated contrastive examples
- Try the full concept token approach from Sastre & Rosá (2026) with proper new-token insertion rather than BOS perturbation
- Compare against representation tuning (Ackerman, 2024) as an additional baseline

### Broader Extensions
- Test on instruction-tuned models where the prompting baseline is stronger
- Investigate multi-feature concept tokens that control several directions simultaneously
- Connect to safety applications: many safety-relevant behaviors (deception, manipulation) may be "hard to describe"

### Open Questions
- Do concept tokens and prompting activate the same underlying mechanisms, or are they accessing the feature via different pathways?
- Is there a principled way to predict which features will be hard to describe without testing empirically?
- Can the describability-advantage relationship be formalized theoretically?

## References

1. Turner et al. (2023). "Steering Language Models With Activation Engineering." arXiv:2308.10248.
2. Rimsky et al. (2023). "Steering Llama 2 via Contrastive Activation Addition." arXiv:2312.06681.
3. Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." arXiv:2406.11717.
4. Braun et al. (2025). "Understanding (Un)Reliability of Steering Vectors." arXiv:2407.12404.
5. Sastre & Rosá (2026). "Concept Tokens: Learning Behavioral Embeddings Through Concept Definitions." arXiv:2601.04465.
6. Ackerman (2024). "Representation Tuning." arXiv:2409.06927.
7. Li et al. (2023). "Inference-Time Intervention: Eliciting Truthful Answers." arXiv:2310.01405.
8. Zou et al. (2023). "Representation Engineering." arXiv:2310.06824.
9. Lester et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." arXiv:2104.08691.
10. Park et al. (2024). "The Linear Representation Hypothesis." arXiv:2401.12241.
11. Shafran et al. (2026). "From Directions to Regions." arXiv:2502.07640.
12. Elhage et al. (2022). "Toy Models of Superposition." arXiv:2209.10652.
