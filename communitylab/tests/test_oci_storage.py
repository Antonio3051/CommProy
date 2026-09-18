"""Tests for the OCI Object Storage client and its local fallback (src/oci/storage)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from src.generators import FAQEntry, GeneratedAsset
from src.oci import (
    DEFAULT_BUCKET,
    AssetStorage,
    StorageSettings,
    UploadResult,
    build_client,
    content_type_for,
)

import oci


@dataclass
class _Response:
    data: Any = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class _Obj:
    name: str


@dataclass
class _Listing:
    objects: list[_Obj]
    next_start_with: str | None = None


class FakeObjectStorage:
    """In-memory stand-in for ``ObjectStorageClient``."""

    def __init__(self, *, fail_with: Exception | None = None, namespace: str = "ns-test") -> None:
        self.fail_with = fail_with
        self.namespace = namespace
        self.objects: dict[tuple[str, str], bytes] = {}
        self.calls: list[dict[str, Any]] = []
        self.namespace_lookups = 0

    def get_namespace(self, **kwargs: Any) -> _Response:
        self.namespace_lookups += 1
        return _Response(data=self.namespace)

    def put_object(
        self, namespace_name: str, bucket_name: str, object_name: str, put_object_body: Any, **kwargs: Any
    ) -> _Response:
        self.calls.append({"namespace": namespace_name, "bucket": bucket_name, "name": object_name, **kwargs})
        if self.fail_with is not None:
            raise self.fail_with
        self.objects[(bucket_name, object_name)] = bytes(put_object_body)
        return _Response(headers={"etag": f"etag-{len(self.objects)}"})

    def list_objects(self, namespace_name: str, bucket_name: str, **kwargs: Any) -> _Response:
        prefix = kwargs.get("prefix", "")
        names = sorted(n for (b, n) in self.objects if b == bucket_name and n.startswith(prefix))
        start = kwargs.get("start")
        if start is None:
            page, nxt = names[:1], (names[1] if len(names) > 1 else None)
        else:
            page, nxt = [n for n in names if n >= start], None
        return _Response(data=_Listing([_Obj(n) for n in page], nxt))

    def get_object(self, namespace_name: str, bucket_name: str, object_name: str, **kwargs: Any) -> _Response:
        return _Response(data=type("Body", (), {"content": self.objects[(bucket_name, object_name)]})())


def service_error(status: int = 404, code: str = "BucketNotFound") -> oci.exceptions.ServiceError:
    return oci.exceptions.ServiceError(status, code, {}, "no such bucket")


@pytest.fixture
def local_settings(tmp_path: Path) -> StorageSettings:
    return StorageSettings(local_dir=tmp_path / "store", force_local=True)


@pytest.fixture
def asset() -> GeneratedAsset:
    return GeneratedAsset(
        channel="faq",
        title="How do I install LangGraph?",
        content=FAQEntry(question="How do I install LangGraph?", answer="pip install langgraph", tags=["pip"]),
        markdown="### How do I install LangGraph?\n\npip install langgraph",
        source_message_ids=("m1", "m2"),
        generated_at=datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC),
    )


class TestSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        for var in ("COMMUNITYLAB_BUCKET", "OCI_NAMESPACE", "OCI_CONFIG_FILE", "OCI_CONFIG_PROFILE"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("COMMUNITYLAB_FORCE_LOCAL", raising=False)
        settings = StorageSettings()
        assert settings.bucket == DEFAULT_BUCKET == "communitylab-activos-marketing"
        assert settings.namespace is None and settings.profile == "DEFAULT"
        assert settings.config_file == "~/.oci/config" and settings.force_local is False
        assert settings.local_dir.parts[-2:] == ("data", DEFAULT_BUCKET)

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("COMMUNITYLAB_BUCKET", "other-bucket")
        monkeypatch.setenv("OCI_NAMESPACE", "ns1")
        monkeypatch.setenv("OCI_CONFIG_PROFILE", "DEV")
        monkeypatch.setenv("COMMUNITYLAB_FORCE_LOCAL", "true")
        settings = StorageSettings()
        assert (settings.bucket, settings.namespace, settings.profile, settings.force_local) == (
            "other-bucket",
            "ns1",
            "DEV",
            True,
        )

    def test_content_types(self):
        assert content_type_for("a/b.md") == "text/markdown; charset=utf-8"
        assert content_type_for("x.JSON") == "application/json; charset=utf-8"
        assert content_type_for("notes.txt") == "text/plain; charset=utf-8"
        assert content_type_for("img.png") == "image/png"
        assert content_type_for("blob") == "application/octet-stream"


class TestBuildClient:
    def test_missing_config_file_returns_none(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        settings = StorageSettings(config_file=str(tmp_path / "nope"), force_local=False)
        with caplog.at_level("WARNING"):
            assert build_client(settings) is None
        assert "falling back to local storage" in caplog.text

    def test_invalid_config_returns_none(self, tmp_path: Path):
        cfg = tmp_path / "config"
        cfg.write_text("[DEFAULT]\nuser=not-an-ocid\n", encoding="utf-8")
        assert build_client(StorageSettings(config_file=str(cfg), force_local=False)) is None

    def test_force_local_skips_oci(self, local_settings: StorageSettings):
        assert build_client(local_settings) is None


class TestLocalFallback:
    def test_no_credentials_saves_under_data(self, local_settings: StorageSettings, asset: GeneratedAsset):
        storage = AssetStorage(local_settings)
        assert storage.is_remote is False
        results = storage.upload_asset(asset)
        assert [r.backend for r in results] == ["local", "local"]
        assert [r.object_name for r in results] == [
            "faq/20260917-120000-how-do-i-install-langgraph.md",
            "faq/20260917-120000-how-do-i-install-langgraph.json",
        ]
        md_path = Path(results[0].location)
        assert md_path.is_relative_to(local_settings.local_dir)
        assert md_path.read_text(encoding="utf-8") == asset.markdown
        payload = json.loads(Path(results[1].location).read_text(encoding="utf-8"))
        assert payload["title"] == asset.title and payload["content"]["tags"] == ["pip"]
        assert results[0].fallback_reason == "OCI credentials not configured"
        assert results[0].size_bytes == len(asset.markdown.encode())

    def test_upload_json_appends_extension_and_serialises(self, local_settings: StorageSettings):
        storage = AssetStorage(local_settings)
        result = storage.upload_json({"when": datetime(2026, 1, 1, tzinfo=UTC), "name": "José"}, "reports/r1")
        assert result.object_name == "reports/r1.json"
        assert result.content_type == "application/json; charset=utf-8"
        assert json.loads(Path(result.location).read_text(encoding="utf-8")) == {
            "when": "2026-01-01 00:00:00+00:00",
            "name": "José",
        }

    def test_upload_file_and_prefix(self, tmp_path: Path):
        settings = StorageSettings(local_dir=tmp_path / "store", force_local=True, prefix="/2026/w38/")
        storage = AssetStorage(settings)
        src = tmp_path / "notes.txt"
        src.write_text("hi", encoding="utf-8")
        result = storage.upload_file(src)
        assert result.object_name == "2026/w38/notes.txt"
        assert (settings.local_dir / "2026" / "w38" / "notes.txt").read_text() == "hi"

    def test_list_and_download_local(self, local_settings: StorageSettings):
        storage = AssetStorage(local_settings)
        assert storage.list_objects() == []
        storage.upload_text("a", "faq/a.md")
        storage.upload_text("b", "linkedin/b.md")
        assert storage.list_objects() == ["faq/a.md", "linkedin/b.md"]
        assert storage.list_objects("faq") == ["faq/a.md"]
        assert storage.download("linkedin/b.md") == b"b"

    def test_path_traversal_is_rejected(self, local_settings: StorageSettings):
        with pytest.raises(ValueError, match="escapes"):
            AssetStorage(local_settings).upload_text("x", "../outside.txt")

    def test_upload_assets_flattens(self, local_settings: StorageSettings, asset: GeneratedAsset):
        results = AssetStorage(local_settings).upload_assets([asset, asset], formats=("md",))
        assert len(results) == 2 and all(r.object_name.endswith(".md") for r in results)


class TestRemoteUpload:
    def test_uploads_to_bucket_with_injected_client(self, asset: GeneratedAsset, tmp_path: Path):
        client = FakeObjectStorage()
        storage = AssetStorage(StorageSettings(local_dir=tmp_path), client=client)
        assert storage.is_remote is True
        results = storage.upload_asset(asset)
        assert all(r.backend == "oci" and r.is_remote for r in results)
        assert results[0].location == "oci://ns-test/communitylab-activos-marketing/" + results[0].object_name
        assert results[0].etag == "etag-1" and results[1].etag == "etag-2"
        assert client.calls[0]["bucket"] == DEFAULT_BUCKET
        assert client.calls[0]["content_type"] == "text/markdown; charset=utf-8"
        assert client.calls[1]["content_type"] == "application/json; charset=utf-8"
        assert client.namespace_lookups == 1
        assert not list(tmp_path.iterdir()), "nothing should be written locally on success"

    def test_configured_namespace_skips_lookup(self, tmp_path: Path):
        client = FakeObjectStorage()
        storage = AssetStorage(StorageSettings(local_dir=tmp_path, namespace="fixed-ns"), client=client)
        result = storage.upload_text("hi", "notes.txt")
        assert result.location == "oci://fixed-ns/communitylab-activos-marketing/notes.txt"
        assert client.namespace_lookups == 0

    def test_service_error_falls_back_to_local(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        client = FakeObjectStorage(fail_with=service_error())
        storage = AssetStorage(StorageSettings(local_dir=tmp_path / "store"), client=client)
        with caplog.at_level("ERROR"):
            result = storage.upload_text("hello", "faq/x.md")
        assert result.backend == "local"
        assert result.fallback_reason is not None and "BucketNotFound" in result.fallback_reason
        assert (tmp_path / "store" / "faq" / "x.md").read_text() == "hello"
        assert "saving faq/x.md locally" in caplog.text

    def test_unexpected_error_falls_back_to_local(self, tmp_path: Path):
        client = FakeObjectStorage(fail_with=ConnectionResetError("reset"))
        storage = AssetStorage(StorageSettings(local_dir=tmp_path / "store"), client=client)
        result = storage.upload_text("hello", "x.txt")
        assert result.backend == "local" and "ConnectionResetError" in (result.fallback_reason or "")

    def test_list_paginates_and_download(self, tmp_path: Path):
        client = FakeObjectStorage()
        storage = AssetStorage(StorageSettings(local_dir=tmp_path, namespace="ns"), client=client)
        storage.upload_text("1", "faq/a.md")
        storage.upload_text("2", "faq/b.md")
        storage.upload_text("3", "linkedin/c.md")
        assert storage.list_objects() == ["faq/a.md", "faq/b.md", "linkedin/c.md"]
        assert storage.list_objects("faq/") == ["faq/a.md", "faq/b.md"]
        assert storage.download("faq/b.md") == b"2"

    def test_upload_result_model(self):
        result = UploadResult(object_name="a", backend="local", location="/tmp/a", size_bytes=1, content_type="t")
        assert result.is_remote is False and result.uploaded_at.tzinfo is not None
        with pytest.raises(ValueError):
            UploadResult(object_name="a", backend="ftp", location="x", size_bytes=1, content_type="t")  # type: ignore[arg-type]
