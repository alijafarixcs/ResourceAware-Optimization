# Resource-Aware LLM Optimization — Runnable Example

This project demonstrates how an AI agent can satisfy a quality target while minimizing cost, latency, tokens, memory, and energy. It uses simulated model, web, and database backends, so the routing behavior is reproducible and requires no API key.

## Run it

The project targets Python 3.11 and has no third-party runtime dependencies.

```bash
pyenv local 3.11
py -3.11 -m venv .venv
.venv/scripts/activate
py -m pip install --upgrade pip
pip install -r requirements.txt
python demo.py
python -m unittest -v
```

The normal demo is deterministic and offline. To connect the `large-cloud` strategy to an OpenAI-compatible chat-completions endpoint, copy `.env.example` to `.env`, fill in the three values, and run `python demo.py --live`. Connection details are read only from `.env`, and `.env` is ignored by Git.

Each result includes the answer, routing decision, measured resource usage, quality score, and a step-by-step decision trace.

## Example decision

A complex architecture request begins with the cheapest viable model. Its output fails the quality gate, so the agent escalates. Long conversation history is summarized before each call. A three-specialist panel can be selected before the large model when its predicted quality is sufficient after accounting for coordination overhead:

```text
analyze: type=text_generation, complexity=complex, quality_target=0.91
route: local-int4 (local, quantization=INT4)
context: 641->... estimated tokens; pruned ... messages
quality-gate: observed=0.64; rejected, escalating
route: small-cloud (remote, quantization=n/a)
quality-gate: observed=0.78; rejected, escalating
route: three-agent-panel (remote, quantization=n/a)
collaboration: three parallel specialists plus synthesis; charged 60 coordination tokens
quality-gate: observed=0.92; accepted
```

The exact trace depends on the request budget. If the remaining budget cannot fund an escalation, the system returns an explicit reduced-capability result instead of crashing or silently overspending.

## Where each optimization appears

| Technique | Example implementation |
|---|---|
| Dynamic model switching | `Router.candidates()` orders local INT4, small-cloud, and large-cloud profiles. |
| Sensitive-output verification | Sensitive requests reserve resources for `output-verifier`; its independent check and usage are recorded separately. |
| Intelligent request routing | `RequestAnalyzer` classifies calculation, fresh-data, organizational-data, sensitive, and text tasks. |
| Adaptive tool selection | Calculations go to `SafeCalculator`; fresh and company queries go to purpose-built providers. |
| Context pruning and summarization | `ContextManager.reduce()` reserves output capacity and replaces older messages with a compact summary marker. |
| Caching | `ResponseCache` reuses accepted non-sensitive answers at zero incremental simulated cost. |
| Output limits | `_truncate()` enforces the remaining token budget. |
| Batch/parallel processing | `handle_batch()` uses a bounded worker pool while preserving request order. |
| Local and cloud models | Profiles identify local vs. remote execution; sensitive work prefers the local path. |
| Quantization/compression | The local profile is labeled INT4 and has lower memory and energy requirements. |
| Budget monitoring | `Router.fits()` checks cumulative cost, latency, tokens, memory, and energy before execution. |
| Proactive prediction | `ResourcePredictor` forecasts usage from a moving window of observed executions. |
| Cost-sensitive collaboration | A three-specialist panel includes synthesis cost, parallel latency, energy, memory, and 60 coordination tokens in its resource profile. |
| Energy-efficient deployment | Energy is part of both the resource profile and the hard request budget. |
| Learned allocation policy | `LearnedPolicy` tracks acceptance rates and feeds them into strategy ranking. |
| Graceful degradation/fallback | An unaffordable or unsuccessful request returns a clear partial/reduced response. |

## Execution flow

```text
Request
  -> classify type and complexity
  -> check response cache
  -> calculate quality target
  -> prune/summarize context for candidate
  -> predict and check resource consumption
  -> select tool or model
  -> execute and measure
  -> optional independent verification for sensitive output
  -> quality gate
       accepted -> cache when safe -> response
       rejected -> escalate while budget remains
       no viable candidate -> graceful fallback
```

## Production adaptation

The optional live path already connects `large-cloud` to an OpenAI-compatible endpoint. Replace the remaining simulated search, database, local-model, and collaboration branches with the relevant clients. Production implementations should also use a real tokenizer, semantic context retrieval, model-based or task-specific evaluators, encrypted/cache-scoped storage, timeouts, retries, observability, and provider rate-limit controls.

The code intentionally never evaluates arbitrary Python: arithmetic is parsed with an allowlisted AST evaluator. Sensitive requests are not cached.
