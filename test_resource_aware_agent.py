import unittest

from resource_aware_agent import (
    Request,
    ResourceAwareAgent,
    ResourceBudget,
    SafeCalculator,
    TaskKind,
)


class ResourceAwareAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ResourceAwareAgent()

    def test_calculation_uses_tool(self) -> None:
        result = self.agent.handle(Request("Calculate (12 + 3) * 4"))
        self.assertEqual(result.strategy, "calculator")
        self.assertEqual(result.answer, "Result: 60")
        self.assertEqual(result.task_kind, TaskKind.CALCULATION)

    def test_calculator_rejects_code(self) -> None:
        with self.assertRaises((ValueError, SyntaxError)):
            SafeCalculator().evaluate("__import__('os').getcwd()")

    def test_repeat_is_cached_at_zero_incremental_cost(self) -> None:
        request = Request("Write a short greeting.")
        first = self.agent.handle(request)
        second = self.agent.handle(request)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.usage.cost_usd, 0)

    def test_sensitive_requests_are_not_cached(self) -> None:
        request = Request("Review this confidential medical note.", contains_sensitive_data=True)
        self.agent.handle(request)
        second = self.agent.handle(request)
        self.assertFalse(second.cache_hit)

    def test_context_is_pruned(self) -> None:
        request = Request(
            "Analyze architecture risks and compare options.",
            context=tuple("x" * 120 for _ in range(30)),
            budget=ResourceBudget(max_tokens=300),
        )
        result = self.agent.handle(request)
        self.assertTrue(any("context:" in step for step in result.trace))

    def test_complex_request_escalates(self) -> None:
        prompt = "Analyze and compare architecture trade-offs, risks, and recommend a strategy. " * 8
        result = self.agent.handle(Request(prompt))
        self.assertEqual(result.strategy, "large-cloud")
        self.assertTrue(any("escalating" in step for step in result.trace))

    def test_tight_budget_degrades_gracefully(self) -> None:
        request = Request(
            "Analyze this confidential contract and all legal risks.",
            contains_sensitive_data=True,
            budget=ResourceBudget(
                max_cost_usd=0, max_latency_ms=1, max_tokens=50, max_memory_mb=10, max_energy_units=0
            ),
        )
        result = self.agent.handle(request)
        self.assertTrue(result.degraded)
        self.assertIn("Unable to complete", result.answer)


if __name__ == "__main__":
    unittest.main()
