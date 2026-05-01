# Assignment writeup

Engineering writeup for the take-home. Setup and run instructions live in [`README.md`](./README.md).

## Why Hugging Face

Considered GitHub API, Wikidata SPARQL, Hugging Face. GitHub is one flat search endpoint — NL→query collapses to "fill in a query string". Wikidata's failure modes are dominated by SPARQL syntax, so the eval becomes "did the model write valid SPARQL". HF has two layered surfaces:

- Hub list endpoints (`/api/{models,datasets,spaces}`) — flat keyword search with structured params.
- Datasets Server (`is-valid` → `splits` → `first-rows`/`rows` → `search`/`filter`/`statistics`) — a small state machine where each call's params depend on the previous response.

A useful answer often takes 2–4 chained calls across both surfaces, which is where naive prompt-to-query pipelines break. Most reads also work without auth.

## Architecture decisions

**Tools, not constrained decoding.** I support OpenAI and Gemini, which have different decoding APIs but both natively support JSON-Schema'd tool calls. Cost: I trust the provider to honour the schema. Upside: one decorator registers a tool for both providers.

**Thin wrappers, not curated abstractions.** I did not wrap the Datasets Server in a "smart" tool that hides endpoint choice. The assignment evaluates NL→query translation, so the tool surface must expose the API surface — picking endpoints for the agent would cheat the eval.

Same tool loop in `hf_agent/app.py` (TUI) and `hf_agent/headless.py`: stream → accumulate → execute → append `tool` messages → re-stream. Caps at 8 rounds (100 for `/loop`).

## Part 1 — break it & harden

12 adversarial queries via `scripts/break_it.py` (ambiguous, conflicting, typo'd, non-English, edge-case). Four failure modes; each fix is in code or schema, not just the prompt.

| ID | Failure | Fix |
|----|---------|-----|
| **F1** | **Schema/API mismatch.** `sort=trending` was in the JSON-Schema enum but the Hub requires `sort=trendingScore` and 400s on the bare word. | **Schema mirrors the API contract.** Enum now lists canonical Hub keys; per-property descriptions map human phrasing → API value at generation time. Same pass added missing params (`author`, `direction`, full sort set). Schema-as-spec, not an alias map — aliases mask the next rename and teach the model the wrong vocabulary. |
| **F2** | **Zero tool calls on ambiguous prompts.** *"show me good ML models"* and *"找一个流行的中文情感分类数据集"* both produced confident answers with **zero** tool calls — silent fallback to training-data recall. | **Grounding enforced via system prompt.** Headless driver requires at least one `hf_hub_search`/`hf_dataset_viewer` call before answering anything dataset/model/space-shaped. Both queries now make 1–2 calls. |
| **F3** | **Silent constraint dropping.** *"text-classification datasets in Spanish, sorted by trending, from before 2020"* returned 2017-era results as if the time filter had been honoured. No HF endpoint supports time-window filtering. | **Constraint reporting required.** Prompt enumerates capabilities the API can't honor (no time filter, no negation, no fuzzy match, English-only tags) and requires the reply to call out anything dropped. |
| **F4** | **Silent spell-correction on opaque IDs.** *"show me the squd dataset"* → silently became `dataset="squad"`. A real project named `squd` would get the wrong answer with no warning. | **No silent spell-correction.** If an ID looks like a typo, search and surface ("Assuming you meant…") rather than substituting silently. |

Non-failure observation: non-English prompts mostly worked. French mapped correctly and answered in French; the Chinese case failed via F2 (no tool calls), not the language.

## Part 1 — what I couldn't fix

Four classes are mitigated, not solved — each rests on a property prompting alone can't remove:

1. **Semantic ambiguity in ranking ("popular", "best").** Prompt forces `sort=downloads` and surfaces the choice. Real fix needs a clarifying turn or per-user prior — neither belongs in a one-shot CLI.
2. **Constraint dropping with no API equivalent.** Prompt forces *reporting*; no LLM produces a time-windowed answer from an API without a time-window filter. Architectural fix: different data source.
3. **Spell-correction on opaque IDs.** "squd"→"squad" works *because* SQuAD is in training data. "qa-v2"→47 candidates does not, and the Hub has no fuzzy-search endpoint. Real fix: embedding index over Hub IDs.
4. **LLM non-determinism in argument generation.** Even with schema enums and a strict prompt, `temperature=1.0` produces different `sort`/`filter` combinations across runs. Honest fix is rigorous eval-and-pin (Part 2), not a better prompt.

Plus **API drift** (out of our control): F1's `trending`→`trendingScore` is exactly the silent contract change that breaks structured-query agents. Pulling the live OpenAPI schema at startup would fix it; we don't.

## Part 2 — eval design

The eval (`evals/`) scores the **generation** of the structured query, not the live response. Rejected scoring against live HF for one reason: provider weather. The Hub returns 5xx and content drift on its own schedule; counting those as model failures burns iteration cycles fixing quota and content problems. Decoupling keeps the eval fast, deterministic modulo provider sampling, and diff-able across runs.

```
evals/
├── cases.py        # 20 NL queries with structured ground truth
├── scorer.py       # predicate DSL + first-call match scoring
├── runner.py       # per-case streaming, RPM throttle, 5xx retry
├── run_eval.py     # CLI: `--models a@15,b@5 --rpm 5 --category typos`
├── rescore.py      # rescore saved results against a changed cases.py
└── results/        # per-model JSON, summary, dated archives
```

Three methodology decisions:

- **One-shot scoring of the first tool call.** Send NL once with the agent's tool schemas; capture the first `ToolCall` only. No execution, no multi-turn loop. Cheap to score, easy to diff. Cost: a model that recovers on round 2 looks like a failure here — called out in pattern #4.
- **Predicate DSL for ground truth.** Bare values match equality; DSL adds `any_of`, `present`, `absent`, `contains`, `startswith`, `regex`. For multi-valid cases (filter-by-task vs query-by-string for QA), `any_of_shapes` accepts any of N alternative arg-shapes. Without it the scorer over-rejected defensible alternatives.
- **Per-model RPM throttling.** Gemma free tier is 5–15 RPM. Runner serializes per-model with `60/rpm` spacing and retries 5xx/429/RESOURCE_EXHAUSTED with backoff. CLI takes `model@rpm` so mixed-tier runs don't bottleneck on the slowest quota.

### Test set (20 hard cases)

| Category | N | What it stresses |
|----------|---|------------------|
| `conflicting` | 3 | Multiple impossibilities at once (no time filter, no negation, no size filter) |
| `ambiguous` | 3 | Compound or fully-subjective ranking |
| `multilingual` | 3 | Russian, Japanese, German — must map to English filter values; one carries embedded negation |
| `multistep` | 3 | First call must be the right *endpoint* (`rows` vs `first-rows`, `search` vs `filter`) |
| `typos` | 3 | Multi-typo, hyphenated, ambiguous |
| `adversarial` | 3 | System-prompt extraction, mixed-legitimate injection, off-platform query |
| `precision` | 2 | Exact endpoint+param combos (`/filter where=`, `/rows offset+length`) |

Conflicting cases are the most adversarial: the right structured query is the one with the impossible constraint *honestly omitted*.

### Model selection

Three models reachable through one Gemini-compatible API key:

| Model | Type | Why |
|-------|------|-----|
| `gemma-4-31b-it` | open-weight, 31B | Largest open model on platform; the prompt-iteration target |
| `gemma-4-26b-a4b-it` | open-weight, 26B MoE | Tests whether the same prompt holds at lower capacity |
| `gemini-3.1-flash-lite-preview` | closed-source | Does the closed-source training advantage hold at distilled size? |

Two open-weight + one closed satisfies the mix; the open pair gives a size sweep separating "model size" from "open-vs-closed".

### Results

20 cases × 3 models, concurrent at 15 RPM. Snapshot in `evals/results/_archive/2026-05-01_3model_20case_v2/`.

| Model | Type | Accuracy | Wall time |
|-------|------|----------|-----------|
| `gemma-4-31b-it` | open-weight (31B) | **95.0%** (19/20) | 155s |
| `gemma-4-26b-a4b-it` | open-weight (26B MoE) | **95.0%** (19/20) | 183s |
| `gemini-3.1-flash-lite-preview` | closed-source | **95.0%** (19/20) | 376s |

All three tied at 95%. **No closed-source advantage at this task** — the capabilities that matter (matching enums, dropping impossible constraints, picking the right endpoint) are ones open-weight checkpoints already learned. Most surprising finding, and the size sweep (31B vs 26B-MoE) didn't separate them either.

Failure inventory — each model has exactly one signature failure:

| Case | 31b | 26b | gemini | What broke |
|------|:---:|:---:|:---:|------------|
| `T02_winogrande_hyphenated` | ✘ | ✘ | ✓ | Both Gemmas trusted user spelling (`dataset="wino-grand"`); closed model searched first |
| `T01_quad_typos` | ✓ | ✓ | ✘ | Closed model emitted hub_search with no `query` on heavy-typo input |

### Patterns

1. **Literal vs fuzzy ID lookup.** "wino-grand" broke both Gemmas (would 404 in production). Flash Lite searched first. Clearest signal that closed-source training paid off — knowing when to disambiguate before committing.
2. **Heavy typos drift closed model into mode collapse.** Inverse of #1: 4-typo `lod minst datset frome hugingface` made Flash Lite emit `hf_hub_search` with no `query`. Both Gemmas extracted "mnist" fine. Open-weight wins one, closed wins the other — symmetric weakness on the typo category.


### What I learned about eval design for structured outputs

1. **Decouple eval from execution.** Scoring generation, not live response, was the biggest force multiplier. Seconds per case, deterministic, doesn't fail on Hub weather.
2. **Ground truth should encode equivalence classes.** Multiple query shapes can be equally correct; a single-answer scorer over-rejects defensible alternatives. `any_of_shapes` makes equivalence explicit and forces you to reason about *actual* equivalence vs accidental closeness.
3. **Hardest cases aren't ambiguous; they're conflicting.** "Popular models" — any sort works. "Transformers but not pytorch" forces admitting the API can't do negation — higher-order capability, better quality signal.
4. **One-shot scoring undercounts agentic recovery.** Worth knowing — and fixing if the eval target shifts from "structured query generation" to "end-to-end task success".
