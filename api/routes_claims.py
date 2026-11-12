"""Claim submission, processing and status routes."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from api.db import Claim, User, engine, get_session
from api.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/claims", tags=["Claims"])


class SubmitClaimRequest(BaseModel):
    policy_number: str
    incident_type: str
    incident_date: str
    incident_description: str
    incident_location: str = ""
    police_report_number: Optional[str] = None
    estimated_amount: float
    vehicle_year: Optional[int] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    documents: list[str] = []


def _claim_to_dict(claim: Claim) -> dict:
    return {
        "id": claim.id,
        "claim_id": claim.claim_id,
        "user_id": claim.user_id,
        "policy_number": claim.policy_number,
        "incident_type": claim.incident_type,
        "incident_date": claim.incident_date,
        "incident_description": claim.incident_description,
        "incident_location": claim.incident_location,
        "estimated_amount": claim.estimated_amount,
        "settlement_amount": claim.settlement_amount,
        "status": claim.status,
        "final_decision": claim.final_decision,
        "fraud_score": claim.fraud_score,
        "fraud_risk_level": claim.fraud_risk_level,
        "agent_call_count": claim.agent_call_count,
        "total_cost_usd": claim.total_cost_usd,
        "processing_time_sec": claim.processing_time_sec,
        "denial_reasons": json.loads(claim.denial_reasons or "[]"),
        "communication_message": claim.communication_message,
        "appeal_instructions": claim.appeal_instructions,
        "agent_outputs": json.loads(claim.agent_outputs or "{}"),
        "pipeline_path": json.loads(claim.pipeline_path or "[]"),
        "error_log": json.loads(claim.error_log or "[]"),
        "created_at": claim.created_at,
        "completed_at": claim.completed_at,
    }


def _persist_pipeline_result(claim: Claim, state: dict, elapsed: float) -> None:
    decision = state.get("final_decision")
    claim.final_decision = (
        decision.value if hasattr(decision, "value") else str(decision or "")
    )
    claim.settlement_amount = state.get("final_amount_usd")
    claim.decided_at = datetime.now(timezone.utc).isoformat()
    claim.decided_by = "AI Agent"

    fraud = state.get("fraud_output")
    if fraud:
        claim.fraud_score = float(getattr(fraud, "fraud_score", 0.0))
        risk = getattr(fraud, "fraud_risk_level", None)
        claim.fraud_risk_level = risk.value if hasattr(risk, "value") else str(risk or "")

    claim.agent_call_count = int(state.get("agent_call_count") or 0)
    claim.total_tokens_used = int(state.get("total_tokens_used") or 0)
    claim.total_cost_usd = float(state.get("total_cost_usd") or 0.0)
    claim.processing_time_sec = (claim.processing_time_sec or 0.0) + elapsed

    trace = state.get("pipeline_trace") or []
    claim.pipeline_path = json.dumps([
        entry.get("agent")
        for entry in trace
        if isinstance(entry, dict) and entry.get("agent")
    ])
    claim.error_log = json.dumps(state.get("error_log") or [])

    agent_outputs = {}
    for key in (
        "intake_output",
        "fraud_output",
        "damage_output",
        "policy_output",
        "settlement_output",
        "communication_output",
    ):
        output = state.get(key)
        if output and hasattr(output, "model_dump"):
            agent_outputs[key] = output.model_dump(mode="json")
    agent_outputs["_trace"] = trace
    claim.agent_outputs = json.dumps(agent_outputs, default=str)

    claim.status = "completed"

    settlement = state.get("settlement_output")
    intake = state.get("intake_output")
    if settlement:
        claim.denial_reasons = json.dumps(settlement.denial_reasons or [])
    elif claim.final_decision == "denied" and intake:
        reasons = list(intake.validation_flags or [])
        if not reasons and intake.intake_notes:
            reasons = [intake.intake_notes]
        claim.denial_reasons = json.dumps(reasons)

    communication = state.get("communication_output")
    if communication:
        claim.communication_message = communication.message
        claim.appeal_instructions = communication.appeal_instructions
    claim.completed_at = datetime.now(timezone.utc).isoformat()


def _mark_failed(claim_pk: int, error: str) -> None:
    try:
        with Session(engine) as session:
            claim = session.get(Claim, claim_pk)
            if not claim:
                return
            claim.status = "failed"
            claim.error_log = json.dumps([error[:500]])
            claim.completed_at = datetime.now(timezone.utc).isoformat()
            session.add(claim)
            session.commit()
    except Exception:
        logger.exception("[claim_pk=%s] could not mark claim as failed", claim_pk)


def _run_pipeline(claim_pk: int) -> None:
    """Run the completed six-agent pipeline in a background thread."""
    try:
        from src.agents.graph import process_claim
        from src.models.state import ClaimInput

        with Session(engine) as session:
            claim = session.get(Claim, claim_pk)
            if not claim:
                return
            claim.status = "processing"
            session.add(claim)
            session.commit()
            session.refresh(claim)

            pipeline_input = ClaimInput(
                claim_id=claim.claim_id,
                policy_number=claim.policy_number,
                claimant_name=claim.claimant_name or "Unknown",
                claimant_email=claim.claimant_email or "unknown@example.com",
                claimant_phone=claim.claimant_phone or "000-000-0000",
                claimant_dob=claim.claimant_dob or "1990-01-01",
                incident_date=claim.incident_date,
                incident_type=claim.incident_type,
                incident_description=claim.incident_description,
                incident_location=claim.incident_location,
                police_report_number=claim.police_report_number,
                estimated_amount=claim.estimated_amount,
                vehicle_year=claim.vehicle_year,
                vehicle_make=claim.vehicle_make,
                vehicle_model=claim.vehicle_model,
                documents=json.loads(claim.documents or "[]"),
                is_appeal=False,
                original_claim_id=None,
            )

            started = time.time()
            result = process_claim(pipeline_input)
            _persist_pipeline_result(claim, result["state"], time.time() - started)
            session.add(claim)
            session.commit()
    except Exception as exc:
        logger.exception("[claim_pk=%s] pipeline failed", claim_pk)
        _mark_failed(claim_pk, f"{type(exc).__name__}: {exc}")


def _spawn_pipeline(claim_pk: int) -> None:
    thread = threading.Thread(
        target=_run_pipeline,
        args=(claim_pk,),
        daemon=True,
        name=f"pipeline-{claim_pk}",
    )
    thread.start()


@router.post("/submit")
def submit_claim(
    body: SubmitClaimRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    claim = Claim(
        claim_id=f"CLM-{uuid.uuid4().hex[:8].upper()}",
        user_id=user.id,
        policy_number=body.policy_number,
        claimant_name=user.username,
        claimant_email=user.email,
        incident_date=body.incident_date,
        incident_type=body.incident_type,
        incident_description=body.incident_description,
        incident_location=body.incident_location,
        police_report_number=body.police_report_number,
        estimated_amount=body.estimated_amount,
        vehicle_year=body.vehicle_year,
        vehicle_make=body.vehicle_make,
        vehicle_model=body.vehicle_model,
        documents=json.dumps(body.documents),
        status="submitted",
    )
    session.add(claim)
    session.commit()
    session.refresh(claim)
    _spawn_pipeline(claim.id)
    return _claim_to_dict(claim)


@router.get("/all")
def get_all_claims(
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    statement = select(Claim)
    if status:
        statement = statement.where(Claim.status == status)
    rows = session.exec(statement.order_by(Claim.created_at.desc()).limit(limit)).all()
    return {"claims": [_claim_to_dict(claim) for claim in rows]}


@router.get("/user/{user_id}")
def get_user_claims(
    user_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if user.role == "user" and user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view other users' claims")
    rows = session.exec(
        select(Claim).where(Claim.user_id == user_id).order_by(Claim.created_at.desc())
    ).all()
    return {"claims": [_claim_to_dict(claim) for claim in rows]}


@router.get("/{claim_id}")
def get_claim(
    claim_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    claim = session.exec(select(Claim).where(Claim.claim_id == claim_id)).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    if user.role == "user" and claim.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your claim")
    return _claim_to_dict(claim)


@router.post("/{claim_id}/process")
def reprocess_claim(
    claim_id: str,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    claim = session.exec(select(Claim).where(Claim.claim_id == claim_id)).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    claim.status = "processing"
    session.add(claim)
    session.commit()
    _spawn_pipeline(claim.id)
    return {"ok": True, "claim_id": claim_id, "status": "processing"}


@router.get("/{claim_id}/status")
def get_claim_status(
    claim_id: str,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    claim = session.exec(select(Claim).where(Claim.claim_id == claim_id)).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return {
        "claim_id": claim.claim_id,
        "status": claim.status,
        "final_decision": claim.final_decision,
    }
