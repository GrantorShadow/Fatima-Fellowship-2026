***

## Evaluation Philosophy

Before running any experiments, one principle must be kept in mind: this is a **base model** (no instruction tuning), so evaluation must use **few-shot prompting**, **perplexity-based scoring (Blank-ppl)**, or **In-Context Light-instruction Prompts (ICLiP)** rather than zero-shot Q&A. The model will otherwise generate continuations rather than answers. All scoring should use greedy decoding for reproducibility. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2503.00812)

The architecture is also unusual: a **Gated DeltaNet hybrid** with 8×(3×DeltaNet→FFN→1×Attention→FFN) blocks, meaning only 1 in 4 layers is true attention — this is a key source of novel failure modes not seen in pure transformer models. [huggingface](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

***

## Master Evaluation Plan

### Phase 0 — Baseline Harness Setup

Before running any dataset, establish:
- **Inference setup**: Use `transformers` + `lm-eval-harness` (EleutherAI) with the base model checkpoint `Qwen/Qwen3.5-4B-Base` [huggingface](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- **Prompting protocol**: 5-shot ICLiP for open-ended; Blank-ppl for MCQ [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2503.00812)
- **Quantization test**: Run the *same* seed evaluation at FP16, Q8, Q6, Q4 to profile degradation curves — the 4B is known to break at Q4 [reddit](https://www.reddit.com/r/LocalLLaMA/comments/1rj8x1q/qwen35_4b_overthinking_to_say_hello/)
- **Decoding config**: Greedy, temperature=0, no penalties (adding `presence_penalty` causes language mixing ) [huggingface](https://huggingface.co/Qwen/Qwen3.5-4B)
- **Metrics**: Accuracy, Exact Match, ROUGE-L, BERTScore, Perplexity, and pass@1 for code

***

### Phase 1 — Language & Multilingual Coverage

**Goal:** Map the 201-language coverage claim against actual comprehension fidelity. [apxml](https://apxml.com/models/qwen35-4b)

| # | Dataset | Task | Languages | Blindspot Targeted |
|---|---|---|---|---|
| 1.1 | **MGSM** | Math word problems | EN, ZH, SW, BN, RU, DE, FR, JA, TH, TE | Multilingual reasoning gap |
| 1.2 | **XCOPA** | Causal commonsense | 11 langs including ET, ID, QU | Low-resource reasoning |
| 1.3 | **IndicQA** | Reading comprehension | HI, BN, TA, TE, ML, MR | Indic language fidelity |
| 1.4 | **FLORES-200** | Translation quality | 200 langs | Tail-language coverage |
| 1.5 | **XStoryCloze** | Story completion | RU, ZH, ES, AR, HI, ID, TE | Narrative coherence cross-lingual |
| 1.6 | **XNLI** | Natural Language Inference | 15 langs | Cross-lingual semantic transfer |
| 1.7 | **AmericasNLI** | NLI | 10 indigenous American languages | Ultra-low-resource extreme |

**Key signal to watch:** Does accuracy in SW, QU, TE drop > 20% vs. EN on identical logical problems? That exposes whether failures are *understanding-side* (token-level misread) or *generation-side*. [arxiv](https://arxiv.org/pdf/2510.27269.pdf)

***

### Phase 2 — Reasoning & Systematic Generalization

**Goal:** Stress the Gated DeltaNet layers' ability to chain logic without full attention at every step.

| # | Dataset | Task Type | Hops | Blindspot Targeted |
|---|---|---|---|---|
| 2.1 | **GSM8K** | Arithmetic word problems | 1–3 | Shallow arithmetic |
| 2.2 | **MATH-500** | Symbolic/algebraic | 3–6 | Deep symbolic reasoning |
| 2.3 | **MuSiQue** | Multi-hop QA | 2–4 | Hop-chaining with document context |
| 2.4 | **ARC-Challenge** | Science reasoning | 2–3 | World knowledge + reasoning |
| 2.5 | **HotpotQA** | Multi-doc reasoning | 2 | Bridging-entity resolution |
| 2.6 | **FOLIO** | First-order logic NLI | — | Formal logical validity |
| 2.7 | **FORGE benchmark** | Systematic generalization (novel combos) | Variable | Out-of-distribution reasoning on unseen contexts  [openreview](https://openreview.net/forum?id=gIvtkwqyub) |
| 2.8 | **BIG-Bench Hard (BBH)** | 23 hard tasks | Mixed | General reasoning ceiling |
| 2.9 | **CLUTRR** | Relational/kinship reasoning | 2–10 | Compositional systematic generalization |

**Key signal:** On CLUTRR, plot accuracy vs. hop-depth. If accuracy collapses after 4 hops, the DeltaNet recurrent memory is leaking context at depth — a specific architectural blindspot. [huggingface](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

***

### Phase 3 — Instruction Following & Format Adherence

**Goal:** Since this is a base model, test its *latent* instruction-following capacity under ICLiP conditions, not post-trained alignment.

| # | Dataset | Constraint Type | Blindspot Targeted |
|---|---|---|---|
| 3.1 | **IFEval** | Format + length + content simultaneously | Multi-constraint adherence |
| 3.2 | **FollowBench** | Progressive constraint stacking (1→5 constraints) | Constraint saturation threshold |
| 3.3 | **InFoBench** | Decomposed instruction following | Partial-constraint failure modes |
| 3.4 | **MT-Bench** (adapted for base) | Multi-turn coherence | Session consistency |
| 3.5 | **Self-consistency check** | Same question, 10 samples | Output variance and over-generation |

**Key signal:** At what constraint count does the base model start dropping constraints? Community reports suggest it struggles at 3+ simultaneous constraints. [reddit](https://www.reddit.com/r/LocalLLaMA/comments/1rj8x1q/qwen35_4b_overthinking_to_say_hello/)

***

### Phase 4 — Factual Grounding & Hallucination

**Goal:** Map entity confusion, temporal drift, and numerical precision failures.

| # | Dataset | Task | Blindspot Targeted |
|---|---|---|---|
| 4.1 | **TriviaQA** | Open-domain fact retrieval | Entity precision |
| 4.2 | **Natural Questions** | Wikipedia-grounded QA | Knowledge boundary |
| 4.3 | **HaluEval** | Hallucination classification | Hallucination type profiling |
| 4.4 | **FActScore** | Fine-grained factuality in long-form | Per-sentence factuality decay |
| 4.5 | **TruthfulQA** | Avoiding false-belief traps | Sycophancy/misconception bias |
| 4.6 | **FEVER** | Fact verification | Binary factual consistency |
| 4.7 | **EntityQuestions** | Rare vs. frequent entity QA | Tail-entity hallucination |
| 4.8 | **Time-Sensitive QA** | Temporal reasoning about events | Date/order confusion |

**Key signal:** On FActScore, does per-sentence factuality decay with generation length? If it does, this exposes the DeltaNet's lossy memory compressing earlier context. [huggingface](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

***

### Phase 5 — Long-Context & Memory

**Goal:** The 262K token native context is a bold claim for a DeltaNet-hybrid with compressed recurrent memory. Probe it systematically. [huggingface](https://huggingface.co/Qwen/Qwen3.5-4B)

| # | Dataset | Context Length | Task | Blindspot Targeted |
|---|---|---|---|---|
| 5.1 | **RULER** | 4K → 128K | Needle-in-haystack, variable tracking | Retrieval precision at depth |
| 5.2 | **LongBench** | 1K–30K | QA, summarization, few-shot | General long-doc understanding |
| 5.3 | **∞Bench** | 100K+ | Book QA, math in context | Ultra-long coherence |
| 5.4 | **BABILong** | Up to 1M tokens | Reasoning over distributed facts | Facts-at-distance retrieval |
| 5.5 | **PassKey Retrieval** | 4K–256K | Single fact buried in noise | Pure retrieval signal/noise ratio |
| 5.6 | **SCROLLS** | Long documents | Summarization + QA | Semantic compression fidelity |

**Probing strategy:** Run RULER at context lengths 4K, 8K, 16K, 32K, 64K, 128K, 256K and plot the accuracy curve. A sharp cliff — likely between 32K and 64K where DeltaNet compression starts dominating — would expose the architecture's memory bottleneck.

***

### Phase 6 — Vision-Language Grounding *(Your Core Research Area)*

**Goal:** Since you work on 3D visual grounding, referring expressions, and VLMs, this phase is highest-yield for your research on whether Qwen3.5-4B is viable as a backbone.

| # | Dataset | Task | Blindspot Targeted |
|---|---|---|---|
| 6.1 | **RefCOCO / RefCOCO+ / RefCOCOg** | Referring expression comprehension | Phrase grounding precision |
| 6.2 | **Visual7W** | Grounded visual QA | Spatial relation accuracy |
| 6.3 | **ScanQA** | 3D scene QA from point clouds | Embodied spatial understanding |
| 6.4 | **SQA3D** | Situated QA in 3D scenes | Egocentric spatial grounding |
| 6.5 | **MMBench** | Structured VLM perception + reasoning | Multi-axis VL evaluation |
| 6.6 | **MMStar** | Hard VLM problems (anti-shortcut) | True multimodal reasoning vs. LM priors |
| 6.7 | **POPE** | Object hallucination | VL hallucination profiling |
| 6.8 | **TextVQA / OCRBench** | OCR + document understanding | Visual text reading accuracy |
| 6.9 | **NExT-QA** | Causal/temporal video QA | Cross-frame temporal grounding |
| 6.10 | **Grounding DINO eval split** | Open-vocab detection grounding | Zero-shot phrase-to-region binding |

**Key signal:** On RefCOCO+, compare "testA" (people-centric) vs. "testB" (object-centric) accuracy — a large gap reveals whether the model's grounding is anchored to salient objects or generalizes to relational expressions. This directly informs your VLN research.

***

### Phase 7 — Code & Structured Output

**Goal:** Base models often inherit coding ability from pre-training data; probe its robustness.

| # | Dataset | Task | Blindspot |
|---|---|---|---|
| 7.1 | **HumanEval+ (EvalPlus)** | Python function synthesis | Correctness on edge cases |
| 7.2 | **MBPP** | Short programming problems | Generalization across problem types |
| 7.3 | **CRUXEval** | Code reasoning (input→output, output→input) | Bidirectional program understanding |
| 7.4 | **SWE-Bench Lite** | Real GitHub issue resolution | End-to-end software engineering |
| 7.5 | **DS-1000** | Data science code (pandas, numpy, sklearn) | Domain-specific library knowledge |

***

### Phase 8 — Safety, Bias & Robustness

**Goal:** Base models lack RLHF guardrails — understanding inherent bias and adversarial fragility is critical before any fine-tuning.

| # | Dataset | Task | Blindspot |
|---|---|---|---|
| 8.1 | **BBQ** | Social bias in QA across 9 categories | Stereotyping under ambiguity |
| 8.2 | **WinoBias / WinoGender** | Gender coreference resolution | Occupational gender bias |
| 8.3 | **AdvGLUE** | Adversarially perturbed NLU | Robustness to word substitution |
| 8.4 | **CheckList** (MFT) | Behavioral testing across linguistic perturbations | Systematic NLU robustness |
| 8.5 | **Multilingual ToxiGen** | Toxic content detection | Cross-lingual safety |
| 8.6 | **BOLD** | Open-ended generation bias profiling | Demographic bias in continuations |

***

## Execution Sequence

Run phases in this order to maximize signal early and avoid wasteful compute:

```
Phase 0 (Setup + Quant profiling)
    ↓
Phase 2 (Reasoning) ──parallel── Phase 1 (Multilingual)
    ↓
Phase 4 (Factual Grounding)
    ↓
Phase 5 (Long-Context)  ──parallel── Phase 6 (Vision-Language)
    ↓
Phase 3 (Instruction Following)
    ↓
Phase 7 (Code) ──parallel── Phase 8 (Safety/Bias)
```

Run Phases 2 and 1 in parallel first because they are compute-light (text-only, MCQ/short-form) and will immediately reveal the highest-impact weaknesses — reasoning depth and multilingual collapse — guiding downstream phase prioritization. [arxiv](https://arxiv.org/pdf/2510.27269.pdf)

***

## Aggregate Reporting Template

After all phases, compile results into a **Blindspot Heatmap** — a matrix of `(Phase × Dataset)` with color-coded accuracy relative to a baseline (e.g., Qwen3-4B or a stronger baseline like Qwen3.5-9B). Any cell below 60% of the baseline score is a **confirmed blindspot** worth targeting in fine-tuning or architecture critique. Given your research goals, Phase 6 results should be cross-referenced against published RefCOCO/ScanQA baselines from recent VLN papers to determine if the 4B base is a viable starting backbone.