"""Dashboard view: sentiment trends, headline metrics and a topic word cloud."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

import altair as alt
import pandas as pd
import streamlit as st

from src.decisions import DecisionReport
from src.interface.data import OutputsRepository, RunRef, enriched_frame, sentiment_trend, topic_frequencies

SENTIMENT_COLORS = {"positive": "#2e7d32", "neutral": "#9e9e9e", "mixed": "#f9a825", "negative": "#c62828"}
LEVEL_COLORS = {"INFO": "#1976d2", "MEDIO": "#f9a825", "ALTO": "#ef6c00", "CRÍTICO": "#c62828"}


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def layout_word_cloud(
    frequencies: Mapping[str, int],
    *,
    width: int = 800,
    height: int = 380,
    max_words: int = 40,
    min_size: int = 14,
    max_size: int = 64,
    seed: int = 7,
) -> pd.DataFrame:
    """Place words on an Archimedean spiral without overlaps (dependency-free word cloud).

    Returns a frame with ``word``, ``count``, ``size`` (font px), ``x`` and ``y``
    (pixel coordinates of the word centre) ready for an Altair text mark.
    """
    items = sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0]))[:max_words]
    if not items:
        return pd.DataFrame(columns=["word", "count", "size", "x", "y"])
    counts = [c for _, c in items]
    lo, hi = min(counts), max(counts)
    rng = random.Random(seed)
    placed: list[tuple[float, float, float, float]] = []  # x0, y0, x1, y1
    rows: list[dict[str, object]] = []

    def try_place(word: str, size: int, theta0: float) -> tuple[float, float] | None:
        box_w, box_h = 0.6 * size * len(word) + 6, size * 1.15
        for step in range(6000):
            theta = theta0 + 0.3 * step
            radius = 0.5 * step
            cx, cy = width / 2 + radius * math.cos(theta), height / 2 + 0.55 * radius * math.sin(theta)
            x0, y0, x1, y1 = cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                continue
            if all(x1 <= px0 or x0 >= px1 or y1 <= py0 or y0 >= py1 for px0, py0, px1, py1 in placed):
                placed.append((x0, y0, x1, y1))
                return cx, cy
        return None

    for word, count in items:
        ratio = 1.0 if hi == lo else (count - lo) / (hi - lo)
        size = round(min_size + (max_size - min_size) * math.sqrt(ratio))
        theta0 = rng.uniform(0, 2 * math.pi)
        # Shrink words that do not fit instead of dropping them.
        while size >= min_size:
            if (spot := try_place(word, size, theta0)) is not None:
                rows.append({"word": word, "count": count, "size": size, "x": spot[0], "y": spot[1]})
                break
            size = int(size * 0.85)
    return pd.DataFrame(rows, columns=["word", "count", "size", "x", "y"])


def headline_metrics(frame: pd.DataFrame, report: DecisionReport | None) -> dict[str, str]:
    """Values for the metric tiles (formatted strings so the view stays dumb)."""
    scored = frame["sentiment_score"].dropna() if not frame.empty else pd.Series(dtype=float)
    negative = int((frame["sentiment_label"] == "negative").sum()) if not frame.empty else 0
    urgent = sum(1 for e in report.escalations if e.is_urgent) if report else 0
    return {
        "Messages analysed": str(len(frame)),
        "Mean sentiment": f"{scored.mean():+.2f}" if not scored.empty else "n/a",
        "Negative messages": f"{negative} ({negative / len(frame):.0%})" if len(frame) else "0",
        "Highest level": report.highest_level.value if report and report.escalations else "—",
        "Urgent escalations": str(urgent),
        "Members at risk": str(sum(1 for r in report.member_risks if r.level != "none")) if report else "0",
    }


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def sentiment_trend_chart(trend: pd.DataFrame) -> alt.LayerChart:
    base = alt.Chart(trend).encode(x=alt.X("period:T", title="Period"))
    line = base.mark_line(point=True, color="#1976d2").encode(
        y=alt.Y("mean_sentiment:Q", title="Mean sentiment", scale=alt.Scale(domain=[-1, 1])),
        tooltip=["period:T", alt.Tooltip("mean_sentiment:Q", format="+.2f"), "messages:Q"],
    )
    bars = base.mark_bar(opacity=0.25, color="#90caf9").encode(
        y=alt.Y("messages:Q", title="Messages", axis=alt.Axis(orient="right"))
    )
    return alt.layer(bars, line).resolve_scale(y="independent").properties(height=280)


def sentiment_distribution_chart(frame: pd.DataFrame) -> alt.Chart:
    counts = frame["sentiment_label"].fillna("unknown").value_counts().rename_axis("label").reset_index(name="count")
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=list(SENTIMENT_COLORS), title=None),
            y=alt.Y("count:Q", title="Messages"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=list(SENTIMENT_COLORS), range=list(SENTIMENT_COLORS.values())),
                legend=None,
            ),
            tooltip=["label:N", "count:Q"],
        )
        .properties(height=280)
    )


def category_chart(frame: pd.DataFrame) -> alt.Chart:
    counts = frame["category"].fillna("unknown").value_counts().rename_axis("category").reset_index(name="count")
    return (
        alt.Chart(counts)
        .mark_bar(color="#5e35b1")
        .encode(
            y=alt.Y("category:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Messages"),
            tooltip=["category:N", "count:Q"],
        )
        .properties(height=280)
    )


def word_cloud_chart(layout: pd.DataFrame, *, width: int = 800, height: int = 380) -> alt.Chart:
    return (
        alt.Chart(layout)
        .mark_text(baseline="middle", align="center", fontWeight="bold")
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, width])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[height, 0])),
            text="word:N",
            size=alt.Size("size:Q", scale=None),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            tooltip=["word:N", "count:Q"],
        )
        .properties(width=width, height=height)
        .configure_view(strokeWidth=0)
    )


# --------------------------------------------------------------------------- #
# View
# --------------------------------------------------------------------------- #
def select_run(repo: OutputsRepository, key: str) -> RunRef | None:
    """Run picker shared by the views; returns ``None`` when nothing is stored yet."""
    runs = repo.list_runs()
    if not runs:
        return None
    labels = {run.label: run for run in runs}
    choice = st.selectbox("Pipeline run", list(labels), key=key)
    return labels[choice]


def render(repo: OutputsRepository) -> None:
    st.title("Community dashboard")
    run = select_run(repo, key="dashboard_run")
    if run is None:
        st.info("No pipeline outputs found yet. Use **Run LangGraph pipeline** in the sidebar to analyse the sample.")
        return

    report = repo.load_report(run)
    frame = enriched_frame(repo.load_enriched(run))

    tiles = headline_metrics(frame, report)
    for column, (label, value) in zip(st.columns(len(tiles)), tiles.items(), strict=True):
        column.metric(label, value)

    if frame.empty:
        st.warning("This run has no stored analysis (it may predate analysis persistence). Charts need enriched data.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Sentiment trend")
            freq = st.radio("Granularity", ["D", "W", "h"], format_func=dict(D="day", W="week", h="hour").get)
            st.altair_chart(sentiment_trend_chart(sentiment_trend(frame, freq)), width="stretch")
        with right:
            st.subheader("Sentiment mix")
            st.altair_chart(sentiment_distribution_chart(frame), width="stretch")

        left, right = st.columns([2, 3])
        with left:
            st.subheader("Categories")
            st.altair_chart(category_chart(frame), width="stretch")
        with right:
            st.subheader("Recurring topics")
            layout = layout_word_cloud(topic_frequencies(frame, report))
            if layout.empty:
                st.caption("No topics extracted.")
            else:
                st.altair_chart(word_cloud_chart(layout), width="stretch")

        with st.expander("Analysed messages", expanded=False):
            st.dataframe(
                frame[
                    [
                        "timestamp",
                        "author",
                        "channel",
                        "sentiment_label",
                        "sentiment_score",
                        "category",
                        "relevance_tier",
                        "content",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    with st.expander("Executive summary", expanded=frame.empty):
        st.markdown(repo.load_summary(run) or report.executive_summary)
