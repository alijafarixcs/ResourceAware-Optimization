"""A dependency-free example of resource-aware LLM/agent orchestration.

The backends in this module are intentionally simulated.  The interesting part is
the control plane: classification, context pruning, routing, budget enforcement,
caching, quality gates, escalation, prediction, and graceful degradation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from threading import Lock
from typing import Iterable


def estimate_tokens(text: str) -> int:
    """A deliberately simple token estimate suitable for routing decisions."""

    return max(1, (len(text) + 3) // 4)


class TaskKind(str, Enum):
    CALCULATION = "calculation"
    FRESH_INFORMATION = "fresh_information"
    ORGANIZATIONAL_DATA = "organizational_data"
    TEXT_GENERATION = "text_generation"
    SENSITIVE_ANALYSIS = "sensitive_analysis"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass(frozen=True)
class ResourceBudget:
    max_cost_usd: float = 0.05
    max_latency_ms: int = 1_000
    max_tokens: int = 2_000
    max_memory_mb: int = 4_096
    max_energy_units: float = 2.0


@dataclass(frozen=True)
class Request:
    prompt: str
    context: tuple[str, ...] = ()
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    required_quality: float | None = None
    contains_sensitive_data: bool = False


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    kind: str
    quality: float
    fixed_cost_usd: float
    latency_ms: int
    memory_mb: int
    energy_units: float
    max_context_tokens: int
    local: bool = False
    quantization: str | None = None


@dataclass
class Usage:
    cost_usd: float = 0.0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    memory_mb: int = 0
    energy_units: float = 0.0

    def add(self, other: "Usage") -> None:
        self.cost_usd += other.cost_usd
        self.latency_ms += other.latency_ms
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.memory_mb = max(self.memory_mb, other.memory_mb)
        self.energy_units += other.energy_units


@dataclass(frozen=True)
class ContextView:
    text: str
    tokens_before: int
    tokens_after: int
    pruned_messages: int


@dataclass
class Result:
    answer: str
    task_kind: TaskKind
    complexity: Complexity
    strategy: str
    quality: float
    usage: Usage
    trace: list[str]
    cache_hit: bool = False
    degraded: bool = False


MODELS = (
    ResourceProfile(
        "local-int4", "model", 0.68, 0.0, 55, 700, 0.10, 700, True, "INT4"
    ),
    ResourceProfile(
        "small-cloud", "model", 0.82, 0.002, 140, 300, 0.08, 2_000
    ),
    ResourceProfile(
        "large-cloud", "model", 0.96, 0.028, 520, 1_600, 0.42, 8_000
    ),
)

TOOLS = {
    TaskKind.CALCULATION: ResourceProfile(
        "calculator", "tool", 0.995, 0.0, 3, 15, 0.002, 200, True
    ),
    TaskKind.FRESH_INFORMATION: ResourceProfile(
        "web-search", "tool", 0.90, 0.004, 180, 80, 0.03, 1_000
    ),
    TaskKind.ORGANIZATIONAL_DATA: ResourceProfile(
        "company-database", "tool", 0.97, 0.001, 35, 60, 0.01, 800, True
    ),
}


class RequestAnalyzer:
    """Cheap heuristics stand in for a small routing classifier."""

    _sensitive = re.compile(r"\b(contract|legal|medical|patient|confidential|pii)\b", re.I)
    _fresh = re.compile(r"\b(latest|current|today|news|weather|price now)\b", re.I)
    _organization = re.compile(
        r"\b(our company|internal|inventory|employee|customer record|database)\b", re.I
    )
    _calculation = re.compile(
        r"(^\s*[\d\s()+\-*/.%]+\s*$)|\b(calculate|compute|multiply|divide|sum)\b",
        re.I,
    )

    def analyze(self, request: Request) -> tuple[TaskKind, Complexity]:
        prompt = request.prompt
        if request.contains_sensitive_data or self._sensitive.search(prompt):
            kind = TaskKind.SENSITIVE_ANALYSIS
        elif self._fresh.search(prompt):
            kind = TaskKind.FRESH_INFORMATION
        elif self._organization.search(prompt):
            kind = TaskKind.ORGANIZATIONAL_DATA
        elif self._calculation.search(prompt):
            kind = TaskKind.CALCULATION
        else:
            kind = TaskKind.TEXT_GENERATION

        score = estimate_tokens(prompt)
        score += len(request.context) * 12
        score += 80 * len(re.findall(r"\b(compare|analyze|trade-off|architecture|risk)\b", prompt, re.I))
        if kind is TaskKind.SENSITIVE_ANALYSIS:
            score += 120
        complexity = (
            Complexity.SIMPLE if score < 80 else Complexity.MEDIUM if score < 220 else Complexity.COMPLEX
        )
        return kind, complexity


class ContextManager:
    """Keeps recent context and converts older messages to a compact summary."""

    def reduce(self, request: Request, max_context_tokens: int) -> ContextView:
        full = "\n".join((*request.context, request.prompt))
        before = estimate_tokens(full)
        # Reserve at least half the request budget for the generated response.
        target = max(40, min(max_context_tokens, request.budget.max_tokens // 2))
        if before <= target:
            return ContextView(full, before, before, 0)

        kept: list[str] = [request.prompt]
        used = estimate_tokens(request.prompt)
        for message in reversed(request.context):
            message_tokens = estimate_tokens(message)
            if used + message_tokens > max(1, target - 25):
                break
            kept.insert(0, message)
            used += message_tokens

        pruned = len(request.context) - (len(kept) - 1)
        summary = f"[Summary: {pruned} older messages omitted; retain prior goals and constraints.]"
        text = "\n".join(([summary] if pruned else []) + kept)
        return ContextView(text, before, estimate_tokens(text), pruned)


class ResponseCache:
    def __init__(self) -> None:
        self._values: dict[str, Result] = {}
        self._lock = Lock()

    @staticmethod
    def key(request: Request) -> str:
        stable = json.dumps(
            {
                "prompt": " ".join(request.prompt.lower().split()),
                "context": request.context,
                "sensitive": request.contains_sensitive_data,
                "quality": request.required_quality,
            },
            sort_keys=True,
        )
        return hashlib.sha256(stable.encode()).hexdigest()

    def get(self, key: str) -> Result | None:
        with self._lock:
            cached = self._values.get(key)
            if cached is None:
                return None
            copy = replace(cached, usage=Usage(), trace=list(cached.trace), cache_hit=True)
            copy.trace.append("cache: reused a previous accepted response")
            return copy

    def put(self, key: str, result: Result) -> None:
        with self._lock:
            self._values[key] = replace(result, trace=list(result.trace))


class ResourcePredictor:
    """Learns average usage per strategy from completed requests."""

    def __init__(self) -> None:
        self._history: dict[str, list[Usage]] = {}
        self._lock = Lock()

    def record(self, strategy: str, usage: Usage) -> None:
        with self._lock:
            self._history.setdefault(strategy, []).append(replace(usage))
            self._history[strategy] = self._history[strategy][-20:]

    def predict(self, profile: ResourceProfile, input_tokens: int) -> Usage:
        with self._lock:
            history = list(self._history.get(profile.name, ()))
        if history:
            count = len(history)
            return Usage(
                cost_usd=sum(x.cost_usd for x in history) / count,
                latency_ms=int(sum(x.latency_ms for x in history) / count),
                input_tokens=input_tokens,
                output_tokens=int(sum(x.output_tokens for x in history) / count),
                memory_mb=max(x.memory_mb for x in history),
                energy_units=sum(x.energy_units for x in history) / count,
            )
        return Usage(
            profile.fixed_cost_usd,
            profile.latency_ms,
            input_tokens,
            min(180, max(20, input_tokens // 2)),
            profile.memory_mb,
            profile.energy_units,
        )


class LearnedPolicy:
    """Tiny feedback policy: successful strategies gain a preference bonus."""

    def __init__(self) -> None:
        self._outcomes: dict[str, list[bool]] = {}
        self._lock = Lock()

    def record(self, strategy: str, accepted: bool) -> None:
        with self._lock:
            self._outcomes.setdefault(strategy, []).append(accepted)
            self._outcomes[strategy] = self._outcomes[strategy][-50:]

    def success_rate(self, strategy: str) -> float:
        with self._lock:
            outcomes = list(self._outcomes.get(strategy, ()))
        return sum(outcomes) / len(outcomes) if outcomes else 0.75


class Router:
    def __init__(self, predictor: ResourcePredictor, policy: LearnedPolicy) -> None:
        self.predictor = predictor
        self.policy = policy

    @staticmethod
    def quality_target(request: Request, kind: TaskKind, complexity: Complexity) -> float:
        if request.required_quality is not None:
            return request.required_quality
        if kind is TaskKind.SENSITIVE_ANALYSIS:
            return 0.93
        return {Complexity.SIMPLE: 0.70, Complexity.MEDIUM: 0.82, Complexity.COMPLEX: 0.91}[complexity]

    @staticmethod
    def fits(usage: Usage, budget: ResourceBudget, already: Usage) -> bool:
        return (
            already.cost_usd + usage.cost_usd <= budget.max_cost_usd
            and already.latency_ms + usage.latency_ms <= budget.max_latency_ms
            and usage.input_tokens + usage.output_tokens <= budget.max_tokens
            and usage.memory_mb <= budget.max_memory_mb
            and already.energy_units + usage.energy_units <= budget.max_energy_units
        )

    def candidates(
        self, kind: TaskKind, complexity: Complexity, sensitive: bool
    ) -> list[ResourceProfile]:
        if kind in TOOLS:
            return [TOOLS[kind]]

        available = list(MODELS)
        if sensitive:
            # Privacy-aware hybrid path: local first; cloud remains an escalation option.
            available.sort(key=lambda p: (not p.local, p.fixed_cost_usd))
        else:
            available.sort(
                key=lambda p: (
                    p.fixed_cost_usd
                    + p.energy_units * 0.01
                    + p.latency_ms / 100_000
                    - self.policy.success_rate(p.name) * 0.001
                )
            )
        return available


class SafeCalculator:
    _ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(self, prompt: str) -> str:
        expression = re.sub(r"(?i)\b(calculate|compute|what is)\b", "", prompt).strip(" ?")
        tree = ast.parse(expression, mode="eval")

        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in self._ops:
                return self._ops[type(node.op)](visit(node.left), visit(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in self._ops:
                return self._ops[type(node.op)](visit(node.operand))
            raise ValueError("Only arithmetic expressions are supported")

        return f"Result: {visit(tree)}"


class SimulatedBackends:
    """Replace these methods with real SDK/API calls in a production system."""

    def __init__(self) -> None:
        self.calculator = SafeCalculator()

    def run(self, profile: ResourceProfile, request: Request, context: ContextView) -> str:
        if profile.name == "calculator":
            return self.calculator.evaluate(request.prompt)
        if profile.name == "web-search":
            return "Fresh-data result: [simulated web search; attach a real search provider here]."
        if profile.name == "company-database":
            return "Organization result: [simulated parameterized database lookup]."
        if profile.name == "local-int4":
            return f"Concise local-model draft for: {request.prompt[:120]}"
        if profile.name == "small-cloud":
            return f"Small-model response for: {request.prompt[:160]}"
        return (
            "Detailed large-model analysis with assumptions, alternatives, risks, and a "
            f"recommended next step for: {request.prompt[:220]}"
        )


class ResourceAwareAgent:
    def __init__(self) -> None:
        self.analyzer = RequestAnalyzer()
        self.context_manager = ContextManager()
        self.cache = ResponseCache()
        self.predictor = ResourcePredictor()
        self.policy = LearnedPolicy()
        self.router = Router(self.predictor, self.policy)
        self.backends = SimulatedBackends()

    @staticmethod
    def _measured_usage(profile: ResourceProfile, context: ContextView, answer: str) -> Usage:
        output_tokens = estimate_tokens(answer)
        return Usage(
            cost_usd=profile.fixed_cost_usd,
            latency_ms=profile.latency_ms,
            input_tokens=context.tokens_after,
            output_tokens=output_tokens,
            memory_mb=profile.memory_mb,
            energy_units=profile.energy_units,
        )

    @staticmethod
    def _observed_quality(
        profile: ResourceProfile, complexity: Complexity, sensitive: bool
    ) -> float:
        penalty = {Complexity.SIMPLE: 0.0, Complexity.MEDIUM: 0.06, Complexity.COMPLEX: 0.14}[complexity]
        if profile.kind == "tool":
            penalty = 0.0
        if sensitive and profile.name != "large-cloud":
            penalty += 0.06
        return max(0.0, profile.quality - penalty)

    @staticmethod
    def _truncate(answer: str, remaining_tokens: int) -> str:
        max_chars = max(0, remaining_tokens * 4)
        if len(answer) <= max_chars:
            return answer
        return answer[: max(0, max_chars - 16)].rstrip() + " …[truncated]"

    def handle(self, request: Request) -> Result:
        key = self.cache.key(request)
        cached = self.cache.get(key)
        if cached:
            return cached

        kind, complexity = self.analyzer.analyze(request)
        target = self.router.quality_target(request, kind, complexity)
        sensitive = request.contains_sensitive_data or kind is TaskKind.SENSITIVE_ANALYSIS
        total = Usage()
        trace = [
            f"analyze: type={kind.value}, complexity={complexity.value}, quality_target={target:.2f}"
        ]
        last_answer = ""
        last_quality = 0.0
        last_strategy = "none"

        for profile in self.router.candidates(kind, complexity, sensitive):
            context = self.context_manager.reduce(request, profile.max_context_tokens)
            predicted = self.predictor.predict(profile, context.tokens_after)
            if not self.router.fits(predicted, request.budget, total):
                trace.append(f"budget: skipped {profile.name}; predicted usage exceeds a limit")
                continue
            trace.append(
                f"route: {profile.name} ({'local' if profile.local else 'remote'}, "
                f"quantization={profile.quantization or 'n/a'})"
            )
            if context.pruned_messages:
                trace.append(
                    f"context: {context.tokens_before}->{context.tokens_after} estimated tokens; "
                    f"pruned {context.pruned_messages} messages"
                )
            try:
                answer = self.backends.run(profile, request, context)
            except (ValueError, SyntaxError, ZeroDivisionError) as exc:
                trace.append(f"execution: {profile.name} failed safely ({exc})")
                self.policy.record(profile.name, False)
                continue

            remaining = request.budget.max_tokens - context.tokens_after
            answer = self._truncate(answer, remaining)
            usage = self._measured_usage(profile, context, answer)
            total.add(usage)
            quality = self._observed_quality(profile, complexity, sensitive)
            accepted = quality >= target
            self.predictor.record(profile.name, usage)
            self.policy.record(profile.name, accepted)
            trace.append(
                f"quality-gate: observed={quality:.2f}; "
                + ("accepted" if accepted else "rejected, escalating")
            )
            last_answer, last_quality, last_strategy = answer, quality, profile.name
            if accepted:
                result = Result(answer, kind, complexity, profile.name, quality, total, trace)
                # Do not persist answers derived from sensitive inputs.
                if not sensitive:
                    self.cache.put(key, result)
                return result

        # Graceful degradation: retain a safe partial result or an explicit template.
        trace.append("fallback: returning reduced-capability response within the budget")
        result = Result(
            last_answer or "Unable to complete within the resource budget; please narrow the request.",
            kind,
            complexity,
            last_strategy,
            last_quality,
            total,
            trace,
            degraded=True,
        )
        return result

    def handle_batch(self, requests: Iterable[Request], workers: int = 4) -> list[Result]:
        """Run independent jobs concurrently; callers keep input ordering."""

        items = list(requests)
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(items)))) as pool:
            return list(pool.map(self.handle, items))


def result_as_dict(result: Result) -> dict[str, object]:
    return {
        "answer": result.answer,
        "decision": {
            "task_kind": result.task_kind.value,
            "complexity": result.complexity.value,
            "strategy": result.strategy,
            "quality": round(result.quality, 3),
            "cache_hit": result.cache_hit,
            "degraded": result.degraded,
        },
        "usage": {
            "cost_usd": round(result.usage.cost_usd, 4),
            "latency_ms": result.usage.latency_ms,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "memory_mb": result.usage.memory_mb,
            "energy_units": round(result.usage.energy_units, 4),
        },
        "trace": result.trace,
    }
