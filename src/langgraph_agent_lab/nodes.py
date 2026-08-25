"""Node implementations for the support-ticket LangGraph workflow.

Every node is a pure state transformer: it receives the current state and returns
only the fields it wants LangGraph to update. LLM-backed nodes have conservative
fallbacks so a provider outage does not create an unsafe action.
"""

from __future__ import annotations

import os
import re
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event

NodeUpdate = dict[str, Any]


class Classification(BaseModel):
    """Structured classification returned by the LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    reason: str = Field(description="Short reason based only on the user's request")


class ToolEvaluation(BaseModel):
    """Structured quality verdict for a tool result."""

    verdict: Literal["success", "needs_retry"]
    reason: str = Field(description="Why the tool result is or is not usable")


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def _response_text(response: object) -> str:
    """Normalize text from LangChain providers with string or block content."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif getattr(block, "text", None):
                parts.append(str(block.text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _fallback_classification(query: str) -> Classification:
    """General fallback used only when the configured provider is unavailable."""
    normalized = query.casefold()
    risky_terms = (
        "refund",
        "delete",
        "cancel",
        "send email",
        "charge",
        "transfer",
        "close account",
    )
    tool_terms = ("lookup", "look up", "status", "track", "search", "find order")
    error_terms = ("timeout", "failure", "failed", "crash", "unavailable", "cannot recover")
    vague_queries = {"help", "fix it", "can you fix it?", "it does not work", "not working"}

    if any(term in normalized for term in risky_terms):
        return Classification(route="risky", reason="Fallback detected a side-effecting action")
    if any(term in normalized for term in tool_terms):
        return Classification(route="tool", reason="Fallback detected an information lookup")
    if normalized.strip(" .!?") in {item.strip(" .!?") for item in vague_queries}:
        return Classification(route="missing_info", reason="Fallback found insufficient context")
    if any(term in normalized for term in error_terms):
        return Classification(route="error", reason="Fallback detected a system failure")
    return Classification(route="simple", reason="Fallback treated the request as informational")


def _approval_is_granted(state: AgentState) -> bool:
    approval = state.get("approval")
    if isinstance(approval, dict):
        return bool(approval.get("approved", False))
    return bool(getattr(approval, "approved", False))


def intake_node(state: AgentState) -> NodeUpdate:
    """Normalize the raw support request."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> NodeUpdate:
    """Classify intent through a real LLM call with a Pydantic output schema."""
    started_at = perf_counter()
    query = state.get("query", "")
    prompt = f"""You route customer-support tickets. Return exactly one route.

Routes:
- risky: an action with side effects, including refunds, deletion, cancellation,
  sending messages, changing accounts, charging, or transferring.
- tool: a read-only lookup, tracking, search, or retrieval request.
- missing_info: vague or incomplete input with too little context to act safely.
- error: a reported technical failure, timeout, crash, or unavailable service.
- simple: an informational question answerable without tools or side effects.

Resolve overlap using this strict priority:
risky > tool > missing_info > error > simple.
Classify semantically; do not rely on scenario identifiers.

User request: {query!r}
"""
    event_type = "completed"
    provider_error = ""
    try:
        classifier = get_llm(temperature=0).with_structured_output(Classification)
        result = classifier.invoke(prompt)
        classification = Classification.model_validate(result)
    except Exception as exc:  # provider/network resilience path
        classification = _fallback_classification(query)
        event_type = "fallback"
        provider_error = f"{type(exc).__name__}: {exc}"

    risk_level = "high" if classification.route == "risky" else "low"
    return {
        "route": classification.route,
        "risk_level": risk_level,
        "classification_reason": classification.reason,
        "events": [
            make_event(
                "classify",
                event_type,
                classification.reason,
                latency_ms=_elapsed_ms(started_at),
                route=classification.route,
                provider_error=provider_error,
            )
        ],
    }


def tool_node(state: AgentState) -> NodeUpdate:
    """Run a deterministic mock tool while preserving production safety gates."""
    started_at = perf_counter()
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))

    if route == "risky" and not _approval_is_granted(state):
        result = "ERROR: risky action blocked because human approval is missing"
    elif route == "error" and attempt < 2:
        result = f"ERROR: transient support tool failure on attempt {attempt}"
    elif route == "risky":
        action = state.get("proposed_action") or state.get("query", "requested action")
        result = f"SUCCESS: approved mock action completed safely: {action}"
    else:
        query = state.get("query", "support request")
        order_match = re.search(r"\b(?:order\s*)?#?([0-9]{3,})\b", query, flags=re.IGNORECASE)
        reference = order_match.group(1) if order_match else "general-request"
        result = f"SUCCESS: mock support lookup completed; reference={reference}; status=processed"

    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                "failed" if result.startswith("ERROR") else "completed",
                result,
                latency_ms=_elapsed_ms(started_at),
                attempt=attempt,
            )
        ],
    }


def evaluate_node(state: AgentState) -> NodeUpdate:
    """Use an LLM-as-judge plus an explicit error-marker safety rule."""
    started_at = perf_counter()
    latest_result = state.get("tool_results", [])[-1] if state.get("tool_results") else ""
    prompt = f"""Judge whether this support-tool result is safe and sufficient for an answer.
Return needs_retry for empty, failed, error, timeout, partial, or unusable results.
Return success only for a complete usable result.

Tool result: {latest_result!r}
"""
    event_type = "completed"
    provider_error = ""
    try:
        judge = get_llm(temperature=0).with_structured_output(ToolEvaluation)
        raw_verdict = judge.invoke(prompt)
        evaluation = ToolEvaluation.model_validate(raw_verdict)
    except Exception as exc:  # provider/network resilience path
        verdict = (
            "needs_retry"
            if not latest_result or "ERROR" in latest_result.upper()
            else "success"
        )
        evaluation = ToolEvaluation(verdict=verdict, reason="Deterministic evaluation fallback")
        event_type = "fallback"
        provider_error = f"{type(exc).__name__}: {exc}"

    # Never allow an LLM judge to turn an explicit tool error into a successful action.
    if not latest_result or "ERROR" in latest_result.upper():
        evaluation.verdict = "needs_retry"
        evaluation.reason = "Explicit tool error requires retry"

    return {
        "evaluation_result": evaluation.verdict,
        "events": [
            make_event(
                "evaluate",
                event_type,
                evaluation.reason,
                latency_ms=_elapsed_ms(started_at),
                verdict=evaluation.verdict,
                provider_error=provider_error,
            )
        ],
    }


def answer_node(state: AgentState) -> NodeUpdate:
    """Generate an LLM response grounded only in workflow state and tool evidence."""
    started_at = perf_counter()
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    prompt = f"""You are a concise customer-support agent.
Answer the request using only the supplied workflow evidence. Do not invent order,
account, payment, or execution details. If a mock tool was used, clearly describe
its result without claiming access to a real production system.

User request: {query!r}
Classified route: {state.get('route', '')!r}
Tool evidence: {tool_results!r}
Approval decision: {approval!r}

Write the final helpful response in the user's language.
"""
    event_type = "completed"
    provider_error = ""
    try:
        answer = _response_text(get_llm(temperature=0.2).invoke(prompt))
        if not answer:
            raise ValueError("LLM returned an empty answer")
    except Exception as exc:  # provider/network resilience path
        evidence = tool_results[-1] if tool_results else "No external tool was required."
        answer = f"Support response for '{query}': {evidence}"
        event_type = "fallback"
        provider_error = f"{type(exc).__name__}: {exc}"

    return {
        "final_answer": answer,
        "messages": [f"assistant:{answer}"],
        "events": [
            make_event(
                "answer",
                event_type,
                "grounded answer generated",
                latency_ms=_elapsed_ms(started_at),
                provider_error=provider_error,
                grounded_on_tool=bool(tool_results),
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> NodeUpdate:
    """Ask for the minimum information needed to continue safely."""
    approval_rejected = state.get("approval") is not None and not _approval_is_granted(state)
    if approval_rejected:
        question = (
            "The proposed action was not approved. Would you like a read-only status check "
            "or a different non-destructive action?"
        )
    else:
        question = (
            "Could you provide the affected product or service, what you expected to happen, "
            "what happened instead, and any relevant reference or error message?"
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": [f"assistant:{question}"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> NodeUpdate:
    """Prepare, but do not execute, a side-effecting action."""
    proposed_action = (
        f"Proposed support action: {state.get('query', '').strip()}. "
        "Execution is blocked until an explicit human approval decision is recorded."
    )
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "prepared",
                "side-effecting action prepared for review",
                risk_level="high",
            )
        ],
    }


def approval_node(state: AgentState) -> NodeUpdate:
    """Record mock approval by default, or pause with LangGraph ``interrupt``."""
    if os.getenv("LANGGRAPH_INTERRUPT", "false").casefold() == "true":
        from langgraph.types import interrupt

        resumed_value = interrupt(
            {
                "kind": "approval_required",
                "question": "Approve this support action?",
                "proposed_action": state.get("proposed_action", ""),
            }
        )
        if isinstance(resumed_value, bool):
            decision = ApprovalDecision(
                approved=resumed_value,
                reviewer="human-reviewer",
                comment="Resumed from LangGraph interrupt",
            )
        else:
            decision = ApprovalDecision.model_validate(resumed_value)
        event_type = "human_decision"
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Automatically approved for deterministic lab execution",
        )
        event_type = "mock_decision"

    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                event_type,
                "risky action approved" if decision.approved else "risky action rejected",
                approved=decision.approved,
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> NodeUpdate:
    """Increment the bounded attempt counter and retain failure evidence."""
    attempt = int(state.get("attempt", 0)) + 1
    latest_result = state.get("tool_results", [])[-1] if state.get("tool_results") else ""
    error = latest_result if "ERROR" in latest_result.upper() else "Retry requested by workflow"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [
            make_event(
                "retry",
                "scheduled",
                f"retry attempt {attempt} recorded",
                attempt=attempt,
                max_attempts=state.get("max_attempts", 3),
            )
        ],
    }


def dead_letter_node(state: AgentState) -> NodeUpdate:
    """Terminate an exhausted workflow with an actionable escalation message."""
    attempt = int(state.get("attempt", 0))
    answer = (
        "We could not complete this request after "
        f"{attempt} attempt(s). It has been moved to manual support review; "
        f"please reference ticket {state.get('scenario_id', 'unknown')}."
    )
    return {
        "final_answer": answer,
        "messages": [f"assistant:{answer}"],
        "events": [
            make_event(
                "dead_letter",
                "escalated",
                "retry limit exhausted; manual review required",
                attempt=attempt,
            )
        ],
    }


def finalize_node(state: AgentState) -> NodeUpdate:
    """Emit the common terminal audit event for every route."""
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                attempts=state.get("attempt", 0),
            )
        ]
    }
