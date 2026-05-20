# LoCoMo Benchmark for keep

Reproduces the [LoCoMo](https://github.com/snap-research/LoCoMo) benchmark for long-term conversational memory using `keep` as the memory backend.

## Results

Evaluated using the standard binary LLM-as-judge methodology (using the `gpt-4o-mini` model),
consistent with published results from other memory systems.
Results are from a single run (not averaged over multiple runs).

| Category | Score | Questions |
|---|---|---|
| Single-hop | 86.2% | 841 |
| Temporal | 68.5% | 321 |
| Multi-hop | 64.2% | 282 |
| Open-domain | 50.0% | 96 |
| **Overall** | **76.2%** | **1540** |


### Stack

| Component | Model | Location |
|---|---|---|
| Embeddings | nomic-embed-text | Local (Ollama) |
| Analysis/summarization | llama3.2:3b | Local (Ollama) |
| Query answering | gpt-4o-mini | OpenAI API |
| Judge | gpt-4o-mini | OpenAI API |

keep's embedding and summarization providers (and their prompts) are user-configurable.
This benchmark used local Ollama models, but keep also supports OpenAI, Anthropic,
and other API providers, as well as the [keep.generalbusiness.ai](https://keep.generalbusiness.ai)
hosted service.

### Comparison with published results

For context, here are publicly reported LoCoMo scores from other memory systems,
sourced from [Memobase](https://github.com/memodb-io/memobase/tree/main/docs/experiments/locomo-benchmark)
and [MemMachine](https://memmachine.ai/blog/2025/09/memmachine-reaches-new-heights-on-locomo/).
Methodologies vary across systems (different models, retrieval strategies,
judge configurations), so these are reference points rather than
strict apples-to-apples comparisons.

| System | Single-hop | Temporal | Multi-hop | Open-domain | Overall |
|---|---|---|---|---|---|
| MemMachine | 93.3 | 72.6 | 80.5 | 64.6 | 84.9 |
| **keep** | **86.2** | **68.5** | **64.2** | **50.0** | **76.2** |
| Memobase v0.0.37 | 70.9 | 85.1 | 46.9 | 77.2 | 75.8 |
| Zep | 74.1 | 79.8 | 66.0 | 67.7 | 75.1 |
| Mem0 | 67.1 | 55.5 | 51.2 | 72.9 | 66.9 |
| LangMem | 62.2 | 23.4 | 47.9 | 71.1 | 58.1 |
| OpenAI | 63.8 | 21.7 | 42.9 | 62.3 | 52.9 |

## Dataset

Download `locomo10.json` from [snap-research/LoCoMo](https://github.com/snap-research/LoCoMo/tree/main/data)
and place it in `dataset/`.

The dataset contains 10 multi-session conversations between character pairs,
with 1,986 QA items across 5 categories:

| Category | Questions | Description |
|---|---|---|
| Single-hop | 841 | Factual recall from a single session |
| Temporal | 321 | Time/date reasoning across sessions |
| Multi-hop | 282 | Synthesizing facts from multiple sessions |
| Open-domain | 96 | Integrating conversation context with world knowledge |
| Adversarial | 446 | Questions about things never discussed (expect refusal) |

**Note on category numbering:** The paper's numbered list (1-5) does not match
the category IDs in the dataset JSON. See [MemMachine's Appendix A](https://memmachine.ai/blog/2025/09/memmachine-reaches-new-heights-on-locomo/#appendix-a)
for the correct mapping.

## Pipeline

### 1. Prepare dataset

```bash
python prep_dataset.py --locomo dataset/locomo10.json --out-dir prepared/
```

Parses the raw LoCoMo JSON into structured files for ingestion:
- `versioned_session_notes.json` — conversation turns grouped by session
- `image_notes.json` — image descriptions with metadata
- `qa_dataset.json` — 1,986 QA items with category labels

### 2. Ingest into keep

```bash
python ingest.py --store stores/run-001 --strategy turns-as-versions --data-dir prepared/
```

Creates an isolated keep store. The `turns-as-versions` strategy models each
conversation session as a versioned document (vstring), with individual turns
as versions. This preserves temporal ordering and enables keep's version-aware
retrieval.

**Prerequisites:** Ollama running (models are pulled automatically on first use).

### 3. Analyze

```bash
keep --store stores/run-001 analyze --all
```

Runs keep's [analysis pipeline](../../docs/KEEP-ANALYZE.md) (using llama3.2:3b via Ollama)
to decompose conversations into searchable parts. Analysis extracts structured facts,
relationships, and temporal markers that improve retrieval quality.

### 4. Query

```bash
python query.py --store stores/run-001 --out results-20260228/run-001_predictions.json \
    --model gpt-4o-mini --deep
```

For each QA item, uses keep's built-in [query prompt template](../../keep/data/system/prompt-agent-query.md) with deep retrieval
(tag-following across related documents). The `--deep` flag enables cross-document
context assembly with a default 3000-token budget.

Supports `--resume-from N` for crash recovery.

### 5. Judge

```bash
python judge_binary.py --predictions results-20260228/run-001_predictions.json \
    --out results-20260228/run-001_judged.json --model gpt-4o-mini
```

Binary LLM-as-judge evaluation using the prompt from the
[Memobase evaluation harness](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/metrics/llm_judge.py).
Adversarial questions are excluded from scoring.

Supports `--resume-from N` for crash recovery.

## Requirements

```
keep-skill>=0.74.0
openai>=1.0
```

These are the defaults used in this benchmark. To use different providers,
configure keep's `embedding` and `summarization` settings (see keep docs).

For local reproduction, Ollama must be running. Models are pulled automatically
on first use.

## Scoring methodology

The **LLM Judge Score** uses gpt-4o-mini to compare each prediction against
the ground truth answer. The judge is instructed to be generous — as long as
the prediction touches on the same topic as the gold answer, it scores 1.
Time-related answers are scored correct if they refer to the same date/period
regardless of format.

The **Overall** score is a weighted average across the four non-adversarial
categories (weighted by question count).

This methodology is consistent with what is used by MemMachine, Memobase, Zep,
Mem0, LangMem, and OpenAI in their published LoCoMo results.
