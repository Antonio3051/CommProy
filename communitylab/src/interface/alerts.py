"""Alerts view: CRÍTICO/ALTO escalations, members at risk and pending actions."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from src.decisions import LEVELS_BY_URGENCY, DecisionReport, Escalation, EscalationLevel, MemberRisk
from src.interface.dashboard import LEVEL_COLORS, select_run
from src.interface.data import OutputsRepository

RISK_ORDER = {"high": 0, "at_risk": 1, "watch": 2, "none": 3}
RISK_BADGES = {"high": "🔴 high", "at_risk": "🟠 at risk", "watch": "🟡 watch", "none": "⚪ none"}


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def urgent_escalations(report: DecisionReport, minimum: EscalationLevel = EscalationLevel.ALTO) -> list[Escalation]:
    """Escalations at or above ``minimum`` (default ALTO), most urgent first."""
    items = [e for e in report.escalations if e.level >= minimum]
    return sorted(items, key=lambda e: (-e.level.rank, -(e.relevance_score or 0.0)))


def flagged_members(report: DecisionReport) -> list[MemberRisk]:
    """Members with at least one R1 signal, highest risk first."""
    items = [r for r in report.member_risks if r.level != "none"]
    return sorted(items, key=lambda r: (RISK_ORDER[r.level], -r.risk_score))


def escalations_frame(items: Sequence[Escalation]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "level": e.level.value,
                "author": e.author,
                "sla_hours": e.sla_hours,
                "requires_response": e.requires_response,
                "relevance": e.relevance_score,
                "sentiment": e.sentiment_score,
                "member_risk": e.member_risk_level,
                "reasons": "; ".join(e.reasons),
                "excerpt": e.excerpt,
                "message_id": e.message_id,
            }
            for e in items
        ]
    )


def members_frame(items: Sequence[MemberRisk]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "level": r.level,
                "author": r.author,
                "risk_score": round(r.risk_score, 2),
                "signals": ", ".join(r.signals),
                "messages": r.message_count,
                "negative": r.negative_count,
                "mean_sentiment": None if r.mean_sentiment is None else round(r.mean_sentiment, 2),
                "days_inactive": r.days_inactive,
                "evidence": " | ".join(r.evidence),
            }
            for r in items
        ]
    )


# --------------------------------------------------------------------------- #
# View
# --------------------------------------------------------------------------- #
def _level_badge(level: EscalationLevel) -> str:
    color = LEVEL_COLORS.get(level.value, "#616161")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;'
        f'font-weight:600;font-size:0.85em">{level.value}</span>'
    )


def render_escalation(escalation: Escalation) -> None:
    with st.container(border=True):
        head, meta = st.columns([3, 2])
        head.markdown(f"{_level_badge(escalation.level)} &nbsp; **{escalation.author}**", unsafe_allow_html=True)
        sla = f"SLA {escalation.sla_hours} h" if escalation.sla_hours is not None else "no SLA"
        response = "awaiting response" if escalation.requires_response else "answered"
        meta.caption(f"{sla} · {response} · member risk: {escalation.member_risk_level} · `{escalation.message_id}`")
        if escalation.excerpt:
            st.markdown(f"> {escalation.excerpt}")
        if escalation.reasons:
            st.caption("Rules fired: " + " · ".join(escalation.reasons))


def render_member(risk: MemberRisk) -> None:
    with st.container(border=True):
        head, meta = st.columns([3, 2])
        head.markdown(f"**{risk.author}** — {RISK_BADGES[risk.level]} (score {risk.risk_score:.2f})")
        inactive = f"{risk.days_inactive} days inactive" if risk.days_inactive is not None else "recently active"
        meta.caption(f"{risk.message_count} msg · {risk.negative_count} negative · {inactive}")
        for line in risk.evidence:
            st.markdown(f"- {line}")


def render(repo: OutputsRepository) -> None:
    st.title("Alerts & escalations")
    run = select_run(repo, key="alerts_run")
    if run is None:
        st.info("No decision reports stored yet. Run the pipeline from the sidebar.")
        return
    report = repo.load_report(run)

    tiles = st.columns(len(LEVELS_BY_URGENCY) + 1)
    for tile, level in zip(tiles, LEVELS_BY_URGENCY, strict=False):
        tile.metric(level.value, str(report.level_counts.get(level.value, 0)))
    tiles[-1].metric("Members at risk", str(len(flagged_members(report))))

    minimum: EscalationLevel = st.radio(
        "Show escalations at or above",
        list(LEVELS_BY_URGENCY),
        index=1,
        horizontal=True,
        format_func=lambda lvl: lvl.value,
        key="alerts_minimum",
    )

    st.subheader("Escalations")
    urgent = urgent_escalations(report, minimum)
    if not urgent:
        st.success(f"No escalations at {minimum.value} or above in this run.")
    else:
        for escalation in urgent:
            render_escalation(escalation)
        with st.expander("As table"):
            st.dataframe(escalations_frame(urgent), width="stretch", hide_index=True)

    st.subheader("Members at risk (R1)")
    members = flagged_members(report)
    if not members:
        st.success("No members flagged by the risk rule.")
    else:
        for risk in members:
            render_member(risk)
        with st.expander("As table"):
            st.dataframe(members_frame(members), width="stretch", hide_index=True)

    st.subheader("Notifications & actions")
    left, right = st.columns(2)
    with left:
        st.markdown("**Notifications**")
        notes = [n for n in report.notifications if n.level >= minimum]
        if not notes:
            st.caption("None at this level.")
        for note in notes:
            with st.expander(f"{note.level.value} · {note.title}"):
                st.caption(f"{note.kind} → {note.target} ({note.audience})")
                st.markdown(note.body)
    with right:
        st.markdown("**Required actions**")
        actions = sorted((a for a in report.actions if a.level >= minimum), key=lambda a: -a.level.rank)
        if not actions:
            st.caption("None at this level.")
        for action in actions:
            due = f" · due in {action.due_hours} h" if action.due_hours is not None else ""
            line = f"- {_level_badge(action.level)} **{action.title.strip()}** — {action.owner}{due}"
            st.markdown(line, unsafe_allow_html=True)
            if action.detail:
                st.caption(action.detail)
