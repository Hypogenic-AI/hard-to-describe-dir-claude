# Research Plan: Manipulation of Hard-to-Describe Directions

## Motivation & Novelty Assessment

### Why This Research Matters
Language models encode many behavioral features as linear directions in their residual streams, but not all are equally accessible via natural language prompting. While it's easy to prompt a model to "be formal" or "rhyme," other features (e.g., reducing AI-sounding output, controlling coherence patterns) resist explicit naming. This limits the practical utility of activation steering and prompt engineering for a large class of model behaviors.

### Gap in Existing Work
Based on the literature review, Braun et al. (2025) demonstrate that steering fails when target behaviors lack coherent directional representations across prompts. Sastre & Rosá (2026) show concept tokens can outperform explicit naming (62% vs 17% for recasting). However, **no study has systematically measured the relationship between a feature's "describability" and the relative advantage of learned token embeddings over prompting**.

### Our Novel Contribution
We test whether the gap between prompting effectiveness and concept-token effectiveness *increases* as features become harder to describe. Specifically, we:
1. Operationalize "describability" as the consistency with which different natural language prompts activate a target direction
2. Show that concept tokens trained via embedding optimization maintain effectiveness even when describability is low
3. Provide the first quantitative evidence that learned tokens offer the most advantage precisely for hard-to-describe features

### Experiment Justification
- **Experiment 1 (Direction Extraction)**: Extract residual stream directions for multiple behavioral features to establish ground truth targets
- **Experiment 2 (Describability Measurement)**: Quantify how well explicit prompts activate each direction — establishes the IV
- **Experiment 3 (Concept Token Training)**: Train token embeddings to reference each direction — tests the core mechanism
- **Experiment 4 (Comparative Evaluation)**: Compare prompting vs. concept tokens — tests the hypothesis that the advantage grows with difficulty

## Research Question
Does the advantage of learned token embeddings over explicit prompting increase as features become harder to describe in natural language, and can concept tokens effectively manipulate directions that prompting cannot?

## Hypothesis Decomposition
1. **H1**: Different features vary in "describability" — the consistency with which prompts activate their corresponding direction
2. **H2**: Concept tokens trained on behavioral examples can activate target directions regardless of describability
3. **H3**: The advantage of concept tokens over prompting is largest for hard-to-describe features

## Proposed Methodology

### Model
- **Pythia-410M** (EleutherAI): Well-supported by TransformerLens, open weights, sufficient capacity to exhibit interesting behavioral directions, manageable for rapid iteration

### Features to Study (Easy → Hard)
1. **Sentiment** (easy): positive vs. negative — well-studied, many natural descriptions
2. **Formality** (easy): formal vs. casual — clearly describable
3. **Verbosity** (medium): verbose vs. concise — somewhat describable
4. **Sycophancy** (medium-hard): agreeable vs. critical — can describe but hard to prompt consistently
5. **AI-sounding** (hard): machine-like vs. human-like text — very hard to articulate what makes text "sound like AI"
6. **Hedging** (hard): cautious/qualified vs. direct assertions — subtle linguistic pattern

### Experimental Steps

#### Step 1: Direction Extraction
- For each feature, create 50 contrastive text pairs (positive vs. negative examples)
- Run through Pythia-410M, extract residual stream activations at layer 12 (middle layer)
- Compute mean difference vector as the feature direction
- Validate: check cosine similarity within positive examples and between positive/negative

#### Step 2: Describability Measurement
- For each feature, write 10 diverse natural language prompts that attempt to invoke it
  - e.g., for sentiment: "Write positively:", "Be optimistic:", "Express happiness:", etc.
  - For AI-sounding: "Write like a human:", "Sound natural:", "Don't be robotic:", etc.
- Measure how consistently each prompt set activates the target direction
- Describability score = mean cosine similarity between prompt-induced activations and the target direction

#### Step 3: Concept Token Training
- Add a new token `<concept_X>` to the tokenizer for each feature
- Training objective: optimize the token embedding so that when prepended to text, the model's residual stream activations shift toward the target direction
- Loss = -cosine_similarity(activation_shift, target_direction) + λ * perplexity_preservation
- Train for 500 steps with Adam, lr=1e-3

#### Step 4: Comparative Evaluation
- For each feature, generate 100 text completions under:
  1. No intervention (baseline)
  2. Best explicit prompt
  3. Concept token prepended
  4. Direct activation addition (CAA baseline)
- Measure: projection onto target direction, behavioral classification accuracy

### Baselines
1. No intervention
2. Explicit prompting (best of 10 prompts)
3. Contrastive Activation Addition (CAA) at optimal layer
4. Concept token (our method)

### Evaluation Metrics
1. **Direction activation**: Mean projection of completions onto target direction
2. **Describability score**: Consistency of prompt-to-direction alignment (cosine sim)
3. **Steering effectiveness**: Behavioral shift relative to baseline
4. **Perplexity preservation**: Change in perplexity on WikiText-2

### Statistical Analysis Plan
- Pearson correlation between describability and prompting effectiveness
- Paired t-tests comparing concept token vs. prompting for each feature
- Two-way analysis: describability level × method interaction
- Bootstrap confidence intervals (n=1000 resamples)

## Expected Outcomes
- Easy features: prompting ≈ concept tokens (both work well)
- Hard features: prompting << concept tokens (tokens still work, prompting fails)
- Strong negative correlation between describability and concept-token advantage

## Timeline
1. Direction extraction & validation: 15 min
2. Describability measurement: 20 min
3. Concept token training: 30 min
4. Comparative evaluation: 20 min
5. Analysis & visualization: 15 min
6. Documentation: 20 min

## Potential Challenges
- Some "hard" features may not be linearly represented → would refute part of hypothesis
- Concept token training may not converge for all features → need careful lr tuning
- Pythia-410M may lack capacity for subtle features → could scale to 1.4B

## Success Criteria
- Clear gradient of describability across features
- Concept tokens show steering effect for ≥5/6 features
- Statistically significant interaction between describability and method effectiveness
