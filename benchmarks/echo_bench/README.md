# ECHO-Bench: Explanation-Grounded Speech Memory

**Status: benchmark design and development fixtures, not a released evaluation dataset.** The examples here are hand-authored text-only fixtures without recorded audio. They are suitable for discussing the data contract, not for reporting ASR accuracy or model generalization.

## Evaluation unit

A session contains multiple complete utterances. Each turn has audio (when collected), a reference transcript, entity annotations, explanation evidence, and expected memory actions. The recognizer receives only earlier turns and the pre-turn memory. New actions become available to subsequent turns. Later explanations may create a revised view of earlier transcripts, while the original recognition outputs remain immutable.

## Tasks

1. Recover an entity spelling from an explanation, including cases where the initial ASR corrupts both the entity and its explanation.
2. Add, update, or delete a hotword, or abstain when no action is justified.
3. Recognize later mentions using accumulated memory without inserting unrelated terms.
4. Revise a historical transcript when later evidence supports a correction, preserving the original output.
5. Evaluate a future end-to-end model that produces a structured thinking segment, a transcript, and actions in one inference procedure.

## Scenario coverage

Include homophones, noisy explanatory phrases, partial character explanations, explicit negation, rare literal spellings, contradictory evidence, topic changes, deletion requests, irrelevant vocabulary, and ordinary speech without entities. Vary the delay between explanation and reuse. Include examples where the first token of a hotword is initially unlikely, not only examples in which its prefix is already recognized correctly.

## Data and splits

Start with development fixtures; collect consented human speech for a separately held-out evaluation set. Synthetic speech can support development or augmentation but must be labeled and evaluated separately. Split by entity and speaker; also hold out explanation templates and phrase families to reduce leakage. Do not tune prompts or bias strengths on the final test sessions. Reference spellings must be confirmed from the intended identity, not inferred solely by normalizing an idiom: literal surnames and deliberate puns are valid counterexamples.

A JSONL session specifies `session_id`, `source`, `modality`, and `turns`. Each turn specifies `turn_id`, `audio`, `reference`, `asr_hypothesis`, and `gold_actions`. Optional span/entity annotations, timestamps, acoustic conditions, and complete post-turn memory states will be added when audio collection begins. `audio: null` is permitted only for the explicitly labeled development fixtures.

## Metrics

Report overall CER/WER, target-entity accuracy, non-target error changes, and incorrect hotword insertions. Score action type and target identity, resulting memory accuracy, and unnecessary updates/deletions. For history edits, measure correction precision, remaining errors, and damage to previously correct spans. Measure whether improvement survives later mentions and whether a deleted term continues to affect decoding. Report latency and compute cost separately. Use session-level uncertainty estimates when the dataset supports them.

## Baselines and ablations

Compare no memory, a supplied static vocabulary, dialogue-history prompting, post-ASR LLM correction, the current ECHO cascade, and a future action-agent variant. An oracle vocabulary is an explicitly labeled upper-reference condition rather than a deployable baseline. For future end-to-end models, compare structured thinking with direct output under matched resources. Ablate explanation interpretation, memory updates, contextual biasing, and history correction individually.

## Development fixtures

[examples/development.jsonl](examples/development.jsonl) illustrates correction and deletion contracts. It does not contain audio, train/test splits, or benchmark scores. The evaluator, data collection, annotation guidelines, and official split release remain future work.
