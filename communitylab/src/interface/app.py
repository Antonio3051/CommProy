"""CommunityLab Streamlit entry point.

Run with ``streamlit run src/interface/app.py`` from the ``communitylab/``
folder. The sidebar switches between the three views and hosts the manual
trigger for the LangGraph pipeline on ``data/sample/messages.json``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# ``streamlit run`` executes this file as a script, so make ``src`` importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interface import alerts, curation, dashboard  # noqa: E402
from src.interface.data import SAMPLE_PATH, OutputsRepository  # noqa: E402
from src.oci import AssetStorage, StorageSettings  # noqa: E402
from src.orchestration.router import ALL_CHANNELS, PipelineOptions  # noqa: E402

PAGES: dict[str, Any] = {
    "Dashboard": dashboard.render,
    "Curation": curation.render,
    "Alerts": alerts.render,
}


@st.cache_resource(show_spinner=False)
def get_repository(force_local: bool) -> OutputsRepository:
    """One storage handle per session (OCI with local fallback, or forced local)."""
    return OutputsRepository(AssetStorage(StorageSettings(force_local=force_local)))


def render_run_controls(repo: OutputsRepository) -> None:
    """Sidebar form that triggers the LangGraph pipeline on the sample file."""
    st.sidebar.subheader("Run pipeline")
    st.sidebar.caption(f"Input: `{SAMPLE_PATH.relative_to(_PROJECT_ROOT)}`")
    has_key = bool(os.environ.get("GROQ_API_KEY"))
    demo = st.sidebar.toggle(
        "Demo mode (mock LLM, no Groq)",
        value=not has_key,
        help="Uses deterministic keyword heuristics instead of Groq so the pipeline runs offline.",
    )
    if not demo and not has_key:
        st.sidebar.warning("GROQ_API_KEY is not set — the run will fail unless demo mode is on.")
    channels = st.sidebar.multiselect("Generators", list(ALL_CHANNELS), default=list(ALL_CHANNELS))
    language = st.sidebar.selectbox("Language", ["en", "es"], index=0)
    force = st.sidebar.checkbox("Force generation", value=demo, help="Run generators even without publishable content.")

    if st.sidebar.button("Run LangGraph pipeline", type="primary", width="stretch"):
        options: PipelineOptions = {
            "channels": list(channels),  # type: ignore[typeddict-item]
            "language": language,
            "force_generate": force,
        }
        with st.spinner("Running ingest → analyze → decide → generate → store…"):
            try:
                summary = repo.run_pipeline(SAMPLE_PATH, demo=demo, options=options)
            except Exception as exc:  # noqa: BLE001 - surfaced to the operator
                st.session_state["last_run_error"] = str(exc)
                st.session_state.pop("last_run", None)
            else:
                st.session_state["last_run"] = summary
                st.session_state.pop("last_run_error", None)
        st.rerun()

    if error := st.session_state.get("last_run_error"):
        st.sidebar.error(f"Pipeline failed: {error}")
    if summary := st.session_state.get("last_run"):
        st.sidebar.success(
            f"Run `{summary['run_id']}` finished: {summary['validated']} messages, "
            f"{len(summary['generated_assets'])} assets, level {summary['highest_level']}."
        )


def main() -> None:
    st.set_page_config(page_title="CommunityLab", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
    st.sidebar.title("🧪 CommunityLab")
    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")

    st.sidebar.divider()
    force_local = st.sidebar.toggle(
        "Read from local fallback only",
        value=os.environ.get("COMMUNITYLAB_FORCE_LOCAL", "").lower() in {"1", "true", "yes"},
        help="Skip OCI even if credentials are present.",
    )
    repo = get_repository(force_local)
    st.sidebar.caption(f"Storage: **{repo.backend}** · `{repo.location}`")

    st.sidebar.divider()
    render_run_controls(repo)

    PAGES[page](repo)


if __name__ == "__main__":
    main()
