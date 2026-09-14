# Resource-Aware LLM Optimization — Runnable Example

This project demonstrates how an AI agent can satisfy a quality target while minimizing cost, latency, tokens, memory, and energy. It uses simulated model, web, and database backends, so the routing behavior is reproducible and requires no API key.

## Run it

Python 3.10 or newer is sufficient; there are no third-party dependencies.

```bash
python demo.py
python -m unittest -v
```

Each result includes the answer, routing decision, measured resource usage, quality score, and a step-by-step decision trace.

## Example decision

A complex architecture request begins with the cheapest viable model. Its output fails the quality gate, so the agent escalates to the large model. Long conversation history is summarized before either call:

```text
analyze: type=text_generation, complexity=complex, quality_target=0.91
route: local-int4 (local, quantization=INT4)
context: 641->... estimated tokens; pruned ... messages
quality-gate: observed=0.54; rejected, escalating
route: small-cloud (remote, quantization=n/a)
quality-gate: observed=0.68; rejected, escalating
route: large-cloud (remote, quantization=n/a)
quality-gate: observed=0.82; ...
```

The exact trace depends on the request budget. If the remaining budget cannot fund an escalation, the system returns an explicit reduced-capability result instead of crashing or silently overspending.

## Where each optimization appears

| Technique | Example implementation |
|---|---|
| Dynamic model switching | `Router.candidates()` orders local INT4, small-cloud, and large-cloud profiles. |
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
| Cost-sensitive collaboration | Independent subtasks use bounded parallel workers; the demo reports sequential vs. parallel latency. |
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
  -> quality gate
       accepted -> cache when safe -> response
       rejected -> escalate while budget remains
       no viable candidate -> graceful fallback
```

## Production adaptation

Keep the control-plane classes and replace `SimulatedBackends.run()` with actual model, search, and database clients. Production implementations should also use a real tokenizer, semantic context retrieval, model-based or task-specific evaluators, encrypted/cache-scoped storage, timeouts, retries, observability, and provider rate-limit controls.

The code intentionally never evaluates arbitrary Python: arithmetic is parsed with an allowlisted AST evaluator. Sensitive requests are not cached.
