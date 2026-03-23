# Literature Review: Manipulation of Hard-to-Describe Directions

## Research Area Overview

This research sits at the intersection of **activation steering** (manipulating internal representations of language models at inference time), **soft prompt tuning** (learning token embeddings to control model behavior), and **mechanistic interpretability** (understanding how features are represented in neural networks). The central hypothesis is that some directions in a model's residual stream are "hard to describe" — they resist manipulation via explicit natural language prompts — but can still be accessed by finetuning a new token embedding to reference those directions.

The field has rapidly matured since 2023. Activation steering via contrastive activation addition (CAA) is now well-established, but significant limitations have emerged: steering is unreliable when the target behavior is not represented by a coherent direction, and different prompt types yield directionally different vectors. Meanwhile, soft prompt tuning has shown that learned embeddings can compactly encode behavioral concepts, sometimes outperforming explicit natural language naming of the same concept.

## Key Papers

### 1. Steering Language Models With Activation Engineering (Turner et al., 2023)
- **arXiv**: 2308.10248
- **Key Contribution**: Introduced Activation Addition (ActAdd) — the foundational method of computing steering vectors from contrastive prompt pairs (e.g., "Love" vs "Hate") and adding them to residual stream activations during inference.
- **Methodology**: Contrast intermediate activations on prompt pairs to compute steering vectors. Add vectors during forward pass for inference-time control.
- **Results**: SOTA on sentiment shift and detoxification using LLaMA-3 and OPT. Lightweight — works with a single pair of data points.
- **Relevance**: The foundational paper for activation steering. Demonstrates that high-level concepts have linear directions in activation space, but relies on the user being able to *name* the contrastive concept in a prompt pair.

### 2. Steering Llama 2 via Contrastive Activation Addition (Rimsky et al., 2023)
- **arXiv**: 2312.06681
- **Key Contribution**: Scaled CAA to behavioral steering on multiple-choice behavioral datasets. Showed that steering vectors from averaged activation differences are effective across many behaviors.
- **Methodology**: Averages differences in residual stream activations between positive/negative behavior examples. Applies vectors at all token positions after the user's prompt.
- **Results**: Significantly alters model behavior, effective over and on top of finetuning and system prompts.
- **Code**: Available; evaluation datasets included.
- **Relevance**: Key baseline method. However, the method requires the user to construct contrastive prompt pairs that explicitly describe the target behavior — the "hard to describe" problem.

### 3. Refusal in Language Models Is Mediated by a Single Direction (Arditi et al., 2024)
- **arXiv**: 2406.11717
- **Key Contribution**: Demonstrated that refusal behavior across 13 chat models is mediated by a one-dimensional subspace in the residual stream. Erasing this direction disables refusal; adding it elicits refusal.
- **519 citations** — highly influential work.
- **Relevance**: Shows that some directions are easy to find and manipulate. The question is: what about directions that don't correspond to a single nameable behavior?

### 4. Understanding (Un)Reliability of Steering Vectors (Braun et al., 2025)
- **arXiv**: 2407.12404
- **Key Contribution**: Systematic study of when steering fails. Finds that (1) no prompt type clearly outperforms others, (2) different prompt types produce directionally different vectors (low cosine similarity), (3) steering is unreliable when the target behavior is not represented by a coherent direction.
- **Key Finding**: "Vector steering is unreliable when the target behavior is not represented by a coherent direction."
- **Relevance**: **Directly supports the research hypothesis.** If a concept can't be described coherently in prompts, contrastive activation addition fails. This motivates the need for an alternative approach (like learned token embeddings) to access these directions.

### 5. Concept Tokens: Learning Behavioral Embeddings Through Concept Definitions (Sastre & Rosá, 2026)
- **arXiv**: 2601.04465
- **Key Contribution**: Adds a new special token to a frozen LLM and learns only its embedding from natural language definitions. The concept token can then be used to steer behavior directionally (assert/negate).
- **Methodology**: Replace concept mentions in a definitional corpus with a new token. Optimize only that token's embedding via cross-entropy loss on the frozen model. Use directionally at inference (e.g., "Do not generate tc" vs. "Generate tc").
- **Key Results**:
  - Hallucination steering: Concept tokens reduce hallucinations comparably to providing full definitions in-context, but with much lower prompt overhead.
  - Recasting: Concept tokens outperform explicit naming of the concept ("recasting") — 62.33% recast rate vs 16.74% for just naming. Crucially, simply naming the technique fails to reliably induce the behavior.
  - Better instruction compliance: Concept tokens preserve follow-up question generation (98.04%) vs in-context definitions (63.07%).
- **Code**: https://github.com/nsuruguay05/concept_tokens
- **Relevance**: **Most directly relevant paper.** Demonstrates that learning a token embedding can capture behavioral concepts more effectively than naming them. This is exactly the mechanism proposed in the research hypothesis — finetuning a new token to reference hard-to-describe directions.

### 6. Representation Tuning (Ackerman, 2024)
- **arXiv**: 2409.06927
- **Key Contribution**: Extended activation steering by permanently finetuning steering vectors into the model using a dual loss function (cosine similarity of activations to target direction + standard token loss).
- **Key Results**: Representation-tuned models show stronger behavioral effects than online steering and generalize better than token-loss-only finetuning.
- **Code**: https://github.com/cma1114/representation_tuning
- **Relevance**: Shows that the bridge between steering vectors and finetuning is productive. Related to our hypothesis of finetuning a token to reference internal directions.

### 7. Inference-Time Intervention: Eliciting Truthful Answers (Li et al., 2023)
- **arXiv**: 2310.01405
- **Methodology**: Shifts activations during inference along truthful directions found in attention heads.
- **Key Result**: Improves Alpaca's TruthfulQA truthfulness from 32.5% to 65.1%.
- **951 citations** — foundational work.
- **Relevance**: Demonstrates that LLMs have internal representations of truthfulness even when producing falsehoods. The internal direction exists but is not easily accessed via prompting.

### 8. Representation Engineering (Zou et al., 2023)
- **arXiv**: 2310.06824
- **Key Contribution**: Introduced "representation engineering" — reading and controlling representations in LLMs by identifying directions corresponding to high-level concepts.
- **Relevance**: Foundational framework for thinking about directions in activation space.

### 9. The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)
- **arXiv**: 2104.08691
- **Key Contribution**: Showed that learning soft prompts (continuous token embeddings) can match full finetuning at scale. As model size increases, prompt tuning becomes more competitive.
- **Relevance**: Foundational method for learning token embeddings. Shows that soft tokens can encode complex task information.

### 10. Prefix-Tuning (Li & Liang, 2021)
- **arXiv**: 2101.00190
- **Key Contribution**: Optimizes continuous prefix vectors prepended to each layer's activations.
- **Relevance**: Alternative approach to learning representations that steer behavior — works at all layers rather than just the embedding layer.

### 11. From Directions to Regions (Shafran et al., 2026)
- **arXiv**: 2502.07640
- **Key Contribution**: Argues that single global directions are insufficient for many concepts. Uses Mixture of Factor Analyzers (MFA) to model activation space as a collection of Gaussian regions with local covariance structure.
- **Key Finding**: MFA outperforms unsupervised baselines and is competitive with supervised methods for concept localization and steering.
- **Relevance**: Directly addresses the limitation of linear direction assumptions. Some "hard to describe" directions may not be single linear directions at all.

### 12. Toy Models of Superposition (Elhage et al., 2022)
- **arXiv**: 2209.10652
- **Key Contribution**: Foundational work on how neural networks represent more features than they have dimensions, through superposition.
- **Relevance**: Provides theoretical grounding for why some features may be hard to isolate via simple contrastive methods — they may be in superposition with other features.

### 13. The Linear Representation Hypothesis (Park et al., 2024)
- **arXiv**: 2401.12241
- **Key Contribution**: Formalizes and tests the hypothesis that high-level concepts are represented as linear directions in activation space.
- **Relevance**: If the hypothesis holds, then hard-to-describe directions still exist as linear features — they just can't be easily referenced via natural language.

## Common Methodologies

1. **Contrastive Activation Addition (CAA)**: Used in Turner 2023, Rimsky 2023, Arditi 2024. Compute steering vectors from contrastive prompt pairs. Limitation: requires describable contrastive concepts.
2. **Soft Prompt / Token Embedding Optimization**: Used in Lester 2021, Li 2021, Sastre 2026. Learn embeddings that steer behavior. Advantage: does not require explicit concept naming.
3. **Representation Tuning**: Ackerman 2024. Finetune models to internalize steering directions via cosine similarity loss.
4. **Sparse Autoencoders (SAEs)**: Used in many recent papers to extract interpretable features from residual streams. Can identify directions that resist natural language description.

## Standard Baselines

- **No intervention**: Unmodified model output
- **Prompting baselines**: Explicit instruction (e.g., "be truthful"), system prompt
- **In-context definitions**: Providing concept definitions in the prompt
- **CAA steering**: Adding contrastive vectors to residual stream
- **SAE feature steering**: Clamping specific SAE features

## Evaluation Metrics

- **TruthfulQA**: Truthfulness and informativeness scores
- **HotpotQA**: Correct/Hallucination/No Answer rates and Precision
- **Behavioral MCQ**: Answer matching on contrastive multiple choice
- **LLM-as-judge**: GPT-4/Claude evaluation of response quality
- **Perplexity**: Preservation of general capabilities (WikiText)

## Datasets in the Literature

- **TruthfulQA** (Lin et al., 2022): Standard truthfulness benchmark, used in ITI, RepTuning
- **HotpotQA** (Yang et al., 2018): Multi-hop QA, used in Concept Tokens
- **StereoSet**: Bias evaluation, used in bias steering papers
- **Custom contrastive pairs**: Various authors generate their own (Rimsky, Turner, Arditi)

## Gaps and Opportunities

1. **No systematic study of "hard to describe" features**: While Braun et al. (2025) show that steering fails when concepts lack coherent directions, no one has systematically categorized which concepts are hard to describe or why.
2. **Concept Tokens haven't been applied to activation-space-identified features**: Sastre & Rosá (2026) learn from natural language definitions. The research hypothesis proposes using SAE-identified or probing-identified features as the target, even when no natural language description suffices.
3. **Bridge between soft prompt tuning and mechanistic interpretability**: The two fields have developed largely independently. Connecting learned token embeddings to specific residual stream directions (via analysis of what directions the learned embedding activates) is unexplored.
4. **Evaluation of steering on non-describable concepts**: All current steering evaluations use concepts that have clear natural language descriptions (honesty, refusal, sentiment). Testing on concepts that resist description would validate the hypothesis.

## Recommendations for Our Experiment

- **Recommended datasets**: TruthfulQA (for truthfulness steering evaluation), HotpotQA (for hallucination steering), custom contrastive pairs for specific features
- **Recommended baselines**: CAA steering, explicit prompting, in-context definitions, unmodified model
- **Recommended metrics**: Behavioral accuracy on steering tasks, cosine similarity between learned token activations and target directions, perplexity preservation
- **Methodological considerations**:
  1. Start with a feature that IS describable (e.g., honesty) and verify that a learned token can steer it
  2. Then identify features via SAEs or probes that lack clean natural language descriptions
  3. Train concept tokens to reference those features using behavioral examples rather than definitions
  4. Compare effectiveness against CAA with various prompt formulations
  5. Use TransformerLens or nnsight for activation analysis
