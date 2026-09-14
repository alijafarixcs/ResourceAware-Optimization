"""Run several requests through the resource-aware agent."""

from __future__ import annotations

import json

from resource_aware_agent import Request, ResourceAwareAgent, ResourceBudget, result_as_dict


def print_result(title: str, result: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(result_as_dict(result), indent=2))


def main() -> None:
    agent = ResourceAwareAgent()

    calculation = Request("Calculate (125 * 8) + 40")
    print_result("Specialized tool", agent.handle(calculation))

    writing = Request("Write a friendly two-sentence welcome message for a new user.")
    print_result("Small/local model", agent.handle(writing))
    print_result("Cached repeat", agent.handle(writing))

    long_context = tuple(
        f"Old project update {i}: scope, stakeholder feedback, and implementation notes."
        for i in range(30)
    )
    complex_request = Request(
        "Analyze the architecture trade-offs and risks, compare three options, and recommend a migration plan.",
        context=long_context,
        budget=ResourceBudget(max_cost_usd=0.04, max_latency_ms=900, max_tokens=500),
    )
    print_result("Context pruning and escalation", agent.handle(complex_request))

    sensitive = Request(
        "Review this confidential contract and identify the major legal risks.",
        contains_sensitive_data=True,
        budget=ResourceBudget(max_cost_usd=0.0, max_latency_ms=100, max_tokens=300),
    )
    print_result("Privacy preference and graceful degradation", agent.handle(sensitive))

    batch = agent.handle_batch(
        [
            Request("Calculate 7 * 9"),
            Request("What is the latest weather news?"),
            Request("Look up our company inventory database"),
        ]
    )
    print("\n=== Parallel batch ===")
    print(json.dumps([result_as_dict(item)["decision"] for item in batch], indent=2))
    print(
        "Estimated sequential latency:", sum(item.usage.latency_ms for item in batch), "ms;",
        "parallel critical path:", max(item.usage.latency_ms for item in batch), "ms",
    )


if __name__ == "__main__":
    main()
