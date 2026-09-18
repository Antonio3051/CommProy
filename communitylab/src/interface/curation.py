"""Curation view: review generated assets and approve or reject them."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from src.interface.data import CurationStatus, OutputsRepository, StoredAsset
from src.orchestration.router import ALL_CHANNELS

CHANNEL_LABELS = {"linkedin": "LinkedIn", "newsletter": "Newsletter", "faq": "FAQ", "testimonial": "Testimonials"}
STATUS_BADGES = {"pending": "🟡 pending", "approved": "🟢 approved", "rejected": "🔴 rejected"}


def filter_assets(
    assets: Sequence[StoredAsset],
    *,
    channels: Sequence[str] = ALL_CHANNELS,
    statuses: Sequence[CurationStatus] = ("pending", "approved", "rejected"),
) -> list[StoredAsset]:
    """Assets matching the selected channels and curation statuses (order preserved)."""
    wanted_channels, wanted_statuses = set(channels), set(statuses)
    return [a for a in assets if a.channel in wanted_channels and a.status in wanted_statuses]


def status_counts(assets: Sequence[StoredAsset]) -> dict[CurationStatus, int]:
    counts: dict[CurationStatus, int] = {"pending": 0, "approved": 0, "rejected": 0}
    for asset in assets:
        counts[asset.status] += 1
    return counts


def _asset_key(asset: StoredAsset) -> str:
    return asset.object_name.replace("/", "_").replace(".", "_")


def render_asset(repo: OutputsRepository, asset: StoredAsset) -> None:
    """One asset card with its Markdown preview and the approve/reject buttons."""
    key = _asset_key(asset)
    header = f"{CHANNEL_LABELS.get(asset.channel, asset.channel)} · {asset.title}"
    with st.container(border=True):
        top_left, top_right = st.columns([4, 1])
        top_left.markdown(f"**{header}**")
        top_left.caption(
            f"{STATUS_BADGES[asset.status]} · {asset.generated_at:%Y-%m-%d %H:%M} UTC · "
            f"{asset.language} · {len(asset.source_message_ids)} source message(s) · `{asset.object_name}`"
        )
        top_right.download_button(
            "Download .md",
            data=asset.markdown,
            file_name=asset.object_name.rsplit("/", 1)[-1].replace(".json", ".md"),
            mime="text/markdown",
            key=f"dl_{key}",
            width="stretch",
        )

        st.markdown(asset.markdown)
        if asset.metadata:
            with st.expander("Metadata"):
                st.json(asset.metadata)

        approve, reject, reset, _ = st.columns([1, 1, 1, 3])
        if approve.button("✅ Approve", key=f"approve_{key}", disabled=asset.status == "approved", width="stretch"):
            repo.set_status(asset.object_name, "approved")
            st.toast(f"Approved: {asset.title}")
            st.rerun()
        if reject.button("❌ Reject", key=f"reject_{key}", disabled=asset.status == "rejected", width="stretch"):
            repo.set_status(asset.object_name, "rejected")
            st.toast(f"Rejected: {asset.title}")
            st.rerun()
        if reset.button("↩ Reset", key=f"reset_{key}", disabled=asset.status == "pending", width="stretch"):
            repo.set_status(asset.object_name, "pending")
            st.rerun()


def render(repo: OutputsRepository) -> None:
    st.title("Content curation")
    assets = repo.load_assets()
    if not assets:
        st.info("No generated assets stored yet. Run the pipeline (with generation enabled) from the sidebar.")
        return

    counts = status_counts(assets)
    tiles = st.columns(4)
    tiles[0].metric("Assets", str(len(assets)))
    tiles[1].metric("Pending", str(counts["pending"]))
    tiles[2].metric("Approved", str(counts["approved"]))
    tiles[3].metric("Rejected", str(counts["rejected"]))

    filter_left, filter_right = st.columns(2)
    channels = filter_left.multiselect(
        "Channels",
        list(ALL_CHANNELS),
        default=list(ALL_CHANNELS),
        format_func=lambda c: CHANNEL_LABELS.get(c, c),
        key="curation_channels",
    )
    statuses = filter_right.multiselect(
        "Status",
        ["pending", "approved", "rejected"],
        default=["pending", "approved", "rejected"],
        key="curation_statuses",
    )

    visible = filter_assets(assets, channels=channels, statuses=statuses)  # type: ignore[arg-type]
    if not visible:
        st.caption("Nothing matches the current filters.")
        return

    tabs = st.tabs([f"{CHANNEL_LABELS.get(c, c)} ({sum(a.channel == c for a in visible)})" for c in channels])
    for tab, channel in zip(tabs, channels, strict=True):
        with tab:
            for asset in (a for a in visible if a.channel == channel):
                render_asset(repo, asset)
