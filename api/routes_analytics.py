"""Analytics routes for completed claim-processing metrics."""
from __future__ import annotations # for Python 3.10-3.11 compatibility

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from api.db import Claim, get_session
from api.security import get_current_user

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


def _load_completed(session: Session) -> list[Claim]:
    return session.exec(select(Claim).where(Claim.status == "completed")).all()

@router.get("/metrics")
def metrics(session: Session = Depends(get_session), _=Depends(get_current_user)):
    """Returns overall metrics and breakdowns for claims, including approval rates, HITL rates, costs, 
    and pipeline paths."""

    all_claims = session.exec(select(Claim)).all()
    completed = [c for c in all_claims if c.status == "completed"]
    total = len(all_claims)

    approved = sum(1 for c in completed if (c.final_decision or "").lower() in ("approved", "approve"))
    approval_rate = (approved / len(completed)) if completed else 0.0
    avg_time = (sum(c.processing_time_sec for c in completed) / len(completed)) if completed else 0.0
    total_cost = sum(c.total_cost_usd for c in all_claims)

    by_type = Counter(c.incident_type or "unknown" for c in all_claims)
    by_decision = Counter((c.final_decision or "pending") for c in completed)

    path_stats = defaultdict(lambda: {"count": 0, "total_cost": 0.0, "total_agents": 0})
    for c in completed:
        path_key = " ->".join(json.loads(c.pipeline_path or "[]")) or "direct"
        s = path_stats[path_key]
        s["count"] += 1
        s["total_cost"] += c.total_cost_usd
        s["total_agents"] += c.agent_call_count

    return {
        "total_claims": total,
        "approval_rate": round(approval_rate, 4),
        "avg_processing_time_sec": round(avg_time, 2),
        "total_cost_usd": round(total_cost, 4),
        "claims_by_type": [{"type": k, "count": v} for k, v in by_type.most_common()],
        "decisions": [{"decision": k, "count": v} for k, v in by_decision.most_common()],
        "pipeline_paths": [
            {
                "path": p,
                "count": s["count"],
                "avg_cost": round(s["total_cost"] / s["count"], 4),
                "avg_agents": round(s["total_agents"] / s["count"], 2),
            }
            for p, s in sorted(path_stats.items(), key=lambda x: -x[1]["count"])
        ],
    }
@router.get("/pipeline")
def pipeline_stats(session: Session = Depends(get_session), _=Depends(get_current_user)):
    """Analyzes the pipeline paths taken by completed claims, including frequency of different paths 
    and agent usage."""

    completed = _load_completed(session)
    agent_calls = Counter()
    for c in completed:
        for agent in json.loads(c.pipeline_path or "[]"):
            agent_calls[agent] += 1
    return {
        "total_completed": len(completed),
        "avg_agents_per_claim": (
            sum(c.agent_call_count for c in completed) / len(completed)
        ) if completed else 0,
        "total_tokens": sum(c.total_tokens_used for c in completed),
        "agent_call_counts": [{"agent": k, "count": v} for k, v in agent_calls.most_common()],
    }


@router.get("/costs")
def costs(
    days: int = Query(30, ge=1, le=365),
    session: Session = Depends(get_session),
    _=Depends(get_current_user),
):
    """"Returns cost-related metrics for claims created in the last `days` days, 
    including total and average costs,"""

    since = datetime.now(timezone.utc) - timedelta(days=days)
    all_claims = session.exec(select(Claim)).all()
    recent = [c for c in all_claims if c.created_at >= since.isoformat()]
    daily = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "claims": 0})
    for c in recent:
        day = c.created_at[:10]
        daily[day]["cost"] += c.total_cost_usd
        daily[day]["tokens"] += c.total_tokens_used
        daily[day]["claims"] += 1
    return {
        "total_cost_usd": round(sum(c.total_cost_usd for c in recent), 4),
        "total_tokens": sum(c.total_tokens_used for c in recent),
        "avg_cost_per_claim": (
            round(sum(c.total_cost_usd for c in recent) / len(recent), 4)
        ) if recent else 0,
        "daily": [{"date": d, **v} for d, v in sorted(daily.items())],
    }


@router.get("/fraud-trends")
def fraud_trends(
    days: int = Query(30, ge=1, le=365),
    session: Session = Depends(get_session),
    _=Depends(get_current_user),
):
    """Analyzes fraud scores and risk levels for claims created in the last `days` days, 
    including average scores, risk level distributions, and daily trends."""
    
    since = datetime.now(timezone.utc) - timedelta(days=days)
    recent = session.exec(select(Claim)).all()
    recent = [c for c in recent if c.created_at >= since.isoformat() and c.fraud_score is not None]
    levels = Counter(c.fraud_risk_level or "unknown" for c in recent)
    daily_avg = defaultdict(list)
    for c in recent:
        daily_avg[c.created_at[:10]].append(c.fraud_score)
    return {
        "avg_fraud_score": (
            round(sum(c.fraud_score for c in recent) / len(recent), 4)
        ) if recent else 0,
        "risk_level_counts": [{"level": k, "count": v} for k, v in levels.most_common()],
        "daily_avg_score": [
            {"date": d, "avg_score": round(sum(v) / len(v), 4)}
            for d, v in sorted(daily_avg.items())
        ],
    }
