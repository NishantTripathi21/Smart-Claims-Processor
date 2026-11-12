"""LangGraph workflow for the completed Smart Claims Processor pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.communication_agent import run_communication_agent
from src.agents.damage_assessor import run_damage_assessor
from src.agents.fraud_crew import run_fraud_crew
from src.agents.intake_agent import run_intake_agent
from src.agents.policy_checker import run_policy_checker
from src.agents.settlement_calculator import run_settlement_calculator
from src.config import get_pipeline_config
from src.models.schemas import ClaimDecision, FraudRiskLevel
from src.models.state import ClaimInput, ClaimsState, initial_state
from src.utils import currency_symbol

logger = logging.getLogger(__name__)


def auto_reject_node(state: ClaimsState) -> dict:
    """Reject a confirmed-fraud claim without further financial processing."""
    claim_id = state["claim"]["claim_id"]
    fraud = state.get("fraud_output")
    score = fraud.fraud_score if fraud else 0.0
    logger.warning("[%s] auto-reject: confirmed fraud (score=%.2f)", claim_id, score)
    return {
        "final_decision": ClaimDecision.AUTO_REJECTED,
        "final_amount_usd": 0.0,
        "pipeline_trace": [{
            "agent": "auto_reject",
            "fraud_score": score,
            "reason": "Confirmed fraud - auto rejected",
        }],
    }


def route_after_intake(
    state: ClaimsState,
) -> Literal["fraud_crew", "communication_agent", "settlement_calculator"]:
    """Route invalid, fast-mode, and normal claims after intake."""
    intake = state.get("intake_output")
    if not intake or not intake.is_valid:
        return "communication_agent"

    fast_mode = get_pipeline_config().get("fast_mode", {})
    if (
        fast_mode.get("enabled", False)
        and float(state["claim"].get("estimated_amount", 0))
        < fast_mode.get("max_amount", 500)
    ):
        return "settlement_calculator"
    return "fraud_crew"


def route_after_fraud(
    state: ClaimsState,
) -> Literal["damage_assessor", "auto_reject"]:
    """Route confirmed fraud to rejection; otherwise continue to damage."""
    fraud = state.get("fraud_output")
    if (
        fraud
        and fraud.fraud_score >= 0.90
        and fraud.fraud_risk_level == FraudRiskLevel.CONFIRMED
    ):
        return "auto_reject"
    return "damage_assessor"


def _build_graph_object() -> StateGraph:
    graph = StateGraph(ClaimsState)
    graph.add_node("intake_agent", run_intake_agent)
    graph.add_node("fraud_crew", run_fraud_crew)
    graph.add_node("damage_assessor", run_damage_assessor)
    graph.add_node("policy_checker", run_policy_checker)
    graph.add_node("settlement_calculator", run_settlement_calculator)
    graph.add_node("auto_reject", auto_reject_node)
    graph.add_node("communication_agent", run_communication_agent)

    graph.add_edge(START, "intake_agent")
    graph.add_conditional_edges("intake_agent", route_after_intake, {
        "fraud_crew": "fraud_crew",
        "communication_agent": "communication_agent",
        "settlement_calculator": "settlement_calculator",
    })
    graph.add_conditional_edges("fraud_crew", route_after_fraud, {
        "damage_assessor": "damage_assessor",
        "auto_reject": "auto_reject",
    })
    graph.add_edge("damage_assessor", "policy_checker")
    graph.add_edge("policy_checker", "settlement_calculator")
    graph.add_edge("settlement_calculator", "communication_agent")
    graph.add_edge("auto_reject", "communication_agent")
    graph.add_edge("communication_agent", END)
    return graph


_compiled_graph = None


def get_compiled_graph():
    """Return the process-wide completed six-agent graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph_object().compile()
    return _compiled_graph


def build_claims_graph():
    """Backwards-compatible alias for callers and tests."""
    return get_compiled_graph()


def _finalize(final_state: dict) -> dict:
    final_state.pop("_guardrails_manager", None)
    return final_state


def _store_to_memory(final_state: dict) -> None:
    """Persist completed outcomes and notable fraud events for future retrieval."""
    try:
        from src.memory.manager import memory

        claim = final_state.get("claim", {})
        claim_id = claim.get("claim_id", "unknown")
        decision = final_state.get("final_decision")
        decision_str = (
            decision.value if hasattr(decision, "value") else str(decision or "unknown")
        )
        fraud = final_state.get("fraud_output")
        fraud_score = float(getattr(fraud, "fraud_score", 0)) if fraud else None

        memory.store_claim_outcome(
            claim_id=claim_id,
            description=claim.get("incident_description", ""),
            metadata={
                "incident_type": claim.get("incident_type", ""),
                "estimated_amount": claim.get("estimated_amount", 0),
                "decision": decision_str,
                "settlement_amount": final_state.get("final_amount_usd", 0),
                "fraud_score": fraud_score,
            },
        )

        if decision_str == ClaimDecision.AUTO_REJECTED.value:
            memory.store_episode(
                claim_id=claim_id,
                narrative=(
                    f"Claim {claim_id} auto-rejected. Fraud score: "
                    f"{fraud_score:.2f}. Description: "
                    f"{claim.get('incident_description', '')[:200]}"
                ),
                event_type="fraud_confirmed",
                metadata={"fraud_score": fraud_score},
            )
    except Exception as exc:
        logger.warning("[%s] memory storage skipped: %s", claim_id, exc)


def process_claim(claim_input: ClaimInput) -> dict:
    """Run the completed pipeline and return its final state."""
    from src.llm import get_token_usage, reset_token_tracking

    claim_id = claim_input["claim_id"]
    graph = get_compiled_graph()
    state = initial_state(claim_input)
    state["execution_start_time"] = datetime.now(timezone.utc).isoformat()
    reset_token_tracking()

    logger.info("[%s] pipeline starting", claim_id)
    try:
        final_state = graph.invoke(state)
    except Exception as exc:
        logger.error("[%s] pipeline crashed: %s", claim_id, exc, exc_info=True)
        state["error_log"] = [f"Pipeline crash: {exc}"]
        state["final_decision"] = ClaimDecision.DENIED
        return {"paused": False, "state": state}

    usage = get_token_usage()
    final_state["total_tokens_used"] = usage["total"]
    final_state["total_cost_usd"] = usage["cost"]
    _store_to_memory(final_state)

    logger.info(
        "[%s] pipeline complete: decision=%s, amount=%s%.2f, tokens=%s, cost=$%.6f",
        claim_id,
        final_state.get("final_decision"),
        currency_symbol(),
        final_state.get("final_amount_usd") or 0,
        usage["total"],
        usage["cost"],
    )
    return {"paused": False, "state": _finalize(final_state)}
