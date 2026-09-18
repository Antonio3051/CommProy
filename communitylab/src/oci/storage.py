"""OCI Object Storage client for generated marketing assets, with a local fallback.

Assets (JSON, Markdown, plain text) go to the bucket
``communitylab-activos-marketing``. When OCI credentials are missing or the
upload fails, the same bytes are written under ``data/`` so local development
and CI never depend on the cloud::

    storage = AssetStorage()                     # picks OCI if ~/.oci/config works, else local
    result = storage.upload_asset(asset)         # -> [UploadResult(backend="oci"|"local", ...)]

Credentials are resolved from the standard OCI config file (``~/.oci/config``
or ``$OCI_CONFIG_FILE``, profile ``$OCI_CONFIG_PROFILE`` / ``DEFAULT``). Set
``COMMUNITYLAB_FORCE_LOCAL=1`` to skip OCI entirely.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

import oci
from src.generators.base import GeneratedAsset

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "communitylab-activos-marketing"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "data" / DEFAULT_BUCKET

Backend = Literal["oci", "local"]

_TEXT_TYPES: dict[str, str] = {
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
}


class ObjectStorageLike(Protocol):
    """Subset of ``oci.object_storage.ObjectStorageClient`` used here (lets tests inject fakes)."""

    def get_namespace(self, **kwargs: Any) -> Any: ...

    def put_object(
        self, namespace_name: str, bucket_name: str, object_name: str, put_object_body: Any, **kwargs: Any
    ) -> Any: ...

    def list_objects(self, namespace_name: str, bucket_name: str, **kwargs: Any) -> Any: ...

    def get_object(self, namespace_name: str, bucket_name: str, object_name: str, **kwargs: Any) -> Any: ...


class StorageSettings(BaseModel):
    """Where and how assets are stored.

    Attributes:
        bucket: Target bucket (``$COMMUNITYLAB_BUCKET`` overrides the default).
        namespace: Object Storage namespace; looked up via the API when ``None``.
        prefix: Optional folder prefix prepended to every object name.
        config_file: OCI config file path.
        profile: Profile inside the config file.
        local_dir: Fallback directory for local saves.
        force_local: Skip OCI and always save locally.
    """

    model_config = ConfigDict(frozen=True)

    bucket: str = Field(default_factory=lambda: os.environ.get("COMMUNITYLAB_BUCKET", DEFAULT_BUCKET), min_length=1)
    namespace: str | None = Field(default_factory=lambda: os.environ.get("OCI_NAMESPACE") or None)
    prefix: str = ""
    config_file: str = Field(default_factory=lambda: os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"))
    profile: str = Field(default_factory=lambda: os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"))
    local_dir: Path = DEFAULT_LOCAL_DIR
    force_local: bool = Field(
        default_factory=lambda: os.environ.get("COMMUNITYLAB_FORCE_LOCAL", "").lower() in ("1", "true", "yes")
    )


class UploadResult(BaseModel):
    """Outcome of one upload (remote or local)."""

    model_config = ConfigDict(frozen=True)

    object_name: str
    backend: Backend
    location: str = Field(description="oci://namespace/bucket/object or a local file path.")
    size_bytes: int = Field(ge=0)
    content_type: str
    etag: str | None = None
    fallback_reason: str | None = Field(default=None, description="Why OCI was skipped, when backend is local.")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_remote(self) -> bool:
        return self.backend == "oci"


def content_type_for(object_name: str) -> str:
    """MIME type from the object extension (UTF-8 for the text formats we emit)."""
    suffix = Path(object_name).suffix.lower()
    if suffix in _TEXT_TYPES:
        return _TEXT_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(object_name)
    return guessed or "application/octet-stream"


def build_client(settings: StorageSettings) -> ObjectStorageLike | None:
    """Create an ``ObjectStorageClient`` from the config file, or ``None`` when credentials are unusable.

    Never raises: every configuration problem is logged and turned into ``None``
    so callers fall back to local storage.
    """
    if settings.force_local:
        logger.info("COMMUNITYLAB_FORCE_LOCAL set; using local storage")
        return None
    try:
        config = oci.config.from_file(file_location=settings.config_file, profile_name=settings.profile)
        oci.config.validate_config(config)
        return oci.object_storage.ObjectStorageClient(config)
    except (
        oci.exceptions.ConfigFileNotFound,
        oci.exceptions.ProfileNotFound,
        oci.exceptions.InvalidConfig,
        oci.exceptions.InvalidKeyFilePath,
        oci.exceptions.InvalidPrivateKey,
        oci.exceptions.MissingPrivateKeyPassphrase,
    ) as exc:
        logger.warning("OCI credentials unavailable (%s: %s); falling back to local storage", type(exc).__name__, exc)
    except Exception as exc:  # noqa: BLE001 - any SDK/IO failure must degrade to local
        logger.warning(
            "Could not initialise OCI client (%s: %s); falling back to local storage", type(exc).__name__, exc
        )
    return None


class AssetStorage:
    """Upload generated assets to OCI Object Storage, saving under ``data/`` when that is not possible.

    Args:
        settings: Bucket/credentials/local-dir configuration.
        client: Pre-built client (tests, or callers using instance principals).
            When omitted, :func:`build_client` is used lazily on first upload.
    """

    def __init__(self, settings: StorageSettings | None = None, *, client: ObjectStorageLike | None = None) -> None:
        self.settings = settings or StorageSettings()
        self._client: ObjectStorageLike | None = client
        self._client_resolved = client is not None
        self._namespace: str | None = self.settings.namespace
        self._fallback_reason: str | None = None

    # ----------------------------------------------------------------- client
    @property
    def client(self) -> ObjectStorageLike | None:
        """The OCI client, built on first access (``None`` in local mode)."""
        if not self._client_resolved:
            self._client = build_client(self.settings)
            self._client_resolved = True
            if self._client is None:
                self._fallback_reason = "OCI credentials not configured"
        return self._client

    @property
    def is_remote(self) -> bool:
        """``True`` when uploads will go to OCI."""
        return self.client is not None

    @property
    def namespace(self) -> str:
        """Object Storage namespace (looked up once via the API when not configured)."""
        if self._namespace is None:
            client = self.client
            if client is None:
                raise RuntimeError("No OCI client available to resolve the namespace")
            self._namespace = str(client.get_namespace().data)
        return self._namespace

    def object_name_for(self, name: str) -> str:
        """Apply the configured prefix and normalise separators."""
        clean = name.replace("\\", "/").lstrip("/")
        prefix = self.settings.prefix.strip("/")
        return f"{prefix}/{clean}" if prefix else clean

    # ---------------------------------------------------------------- uploads
    def upload_bytes(self, data: bytes, object_name: str, *, content_type: str | None = None) -> UploadResult:
        """Upload raw bytes; falls back to a local file if OCI is unavailable or the call fails."""
        object_name = self.object_name_for(object_name)
        content_type = content_type or content_type_for(object_name)
        client = self.client
        if client is None:
            return self._save_local(data, object_name, content_type, reason=self._fallback_reason)
        try:
            response = client.put_object(
                namespace_name=self.namespace,
                bucket_name=self.settings.bucket,
                object_name=object_name,
                put_object_body=data,
                content_type=content_type,
            )
        except (oci.exceptions.ServiceError, oci.exceptions.RequestException) as exc:
            reason = f"OCI upload failed ({type(exc).__name__}: {exc})"
            logger.error("%s; saving %s locally", reason, object_name)
            return self._save_local(data, object_name, content_type, reason=reason)
        except Exception as exc:  # noqa: BLE001 - never lose an asset because of the transport
            reason = f"Unexpected error during OCI upload ({type(exc).__name__}: {exc})"
            logger.exception("%s; saving %s locally", reason, object_name)
            return self._save_local(data, object_name, content_type, reason=reason)

        etag = response.headers.get("etag")
        logger.info(
            "Uploaded %s (%d bytes) to oci://%s/%s", object_name, len(data), self.namespace, self.settings.bucket
        )
        return UploadResult(
            object_name=object_name,
            backend="oci",
            location=f"oci://{self.namespace}/{self.settings.bucket}/{object_name}",
            size_bytes=len(data),
            content_type=content_type,
            etag=etag,
        )

    def upload_text(self, text: str, object_name: str, *, content_type: str | None = None) -> UploadResult:
        """Upload a UTF-8 text/Markdown document."""
        return self.upload_bytes(text.encode("utf-8"), object_name, content_type=content_type)

    def upload_json(self, payload: Any, object_name: str, *, indent: int | None = 2) -> UploadResult:
        """Serialise ``payload`` (any JSON-able object) and upload it."""
        if not object_name.lower().endswith(".json"):
            object_name += ".json"
        body = json.dumps(payload, indent=indent, ensure_ascii=False, default=str)
        return self.upload_text(body, object_name, content_type=_TEXT_TYPES[".json"])

    def upload_file(self, path: Path | str, object_name: str | None = None) -> UploadResult:
        """Upload an existing local file (object name defaults to the file name)."""
        path = Path(path)
        return self.upload_bytes(path.read_bytes(), object_name or path.name)

    def upload_asset(
        self,
        asset: GeneratedAsset,
        *,
        formats: Sequence[Literal["md", "json", "txt"]] = ("md", "json"),
    ) -> list[UploadResult]:
        """Store one generated asset in each requested format under ``<channel>/``."""
        results: list[UploadResult] = []
        for fmt in formats:
            name = asset.filename(fmt)
            if fmt == "json":
                results.append(self.upload_json(asset.to_record(), name))
            else:
                results.append(self.upload_text(asset.markdown, name))
        return results

    def upload_assets(
        self,
        assets: Iterable[GeneratedAsset],
        *,
        formats: Sequence[Literal["md", "json", "txt"]] = ("md", "json"),
    ) -> list[UploadResult]:
        """:meth:`upload_asset` for many assets, flattened."""
        return [r for asset in assets for r in self.upload_asset(asset, formats=formats)]

    # ------------------------------------------------------------------ reads
    def list_objects(self, prefix: str = "") -> list[str]:
        """Object names under ``prefix`` (remote bucket or local folder)."""
        prefix = self.object_name_for(prefix) if prefix else self.settings.prefix.strip("/")
        client = self.client
        if client is None:
            root = self.settings.local_dir
            if not root.exists():
                return []
            names = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
            return sorted(n for n in names if n.startswith(prefix))
        names: list[str] = []
        start: str | None = None
        while True:
            kwargs: dict[str, Any] = {"prefix": prefix} if prefix else {}
            if start:
                kwargs["start"] = start
            response = client.list_objects(self.namespace, self.settings.bucket, **kwargs)
            names.extend(obj.name for obj in response.data.objects)
            start = response.data.next_start_with
            if not start:
                return sorted(names)

    def download(self, object_name: str) -> bytes:
        """Fetch an object's bytes from the bucket, or from the local folder in local mode."""
        object_name = self.object_name_for(object_name)
        client = self.client
        if client is None:
            return self._local_path(object_name).read_bytes()
        response = client.get_object(self.namespace, self.settings.bucket, object_name)
        return bytes(response.data.content)

    # ------------------------------------------------------------------ local
    def _local_path(self, object_name: str) -> Path:
        root = self.settings.local_dir.resolve()
        target = (root / object_name).resolve()
        if root not in target.parents:
            raise ValueError(f"Object name escapes the local storage directory: {object_name!r}")
        return target

    def _save_local(self, data: bytes, object_name: str, content_type: str, *, reason: str | None) -> UploadResult:
        path = self._local_path(object_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.info("Saved %s (%d bytes) locally at %s", object_name, len(data), path)
        return UploadResult(
            object_name=object_name,
            backend="local",
            location=str(path),
            size_bytes=len(data),
            content_type=content_type,
            fallback_reason=reason,
        )
