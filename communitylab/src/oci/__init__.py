"""Oracle Cloud Infrastructure integrations (Object Storage for generated assets)."""

from src.oci.storage import (
    DEFAULT_BUCKET,
    DEFAULT_LOCAL_DIR,
    AssetStorage,
    Backend,
    ObjectStorageLike,
    StorageSettings,
    UploadResult,
    build_client,
    content_type_for,
)

__all__ = [
    "DEFAULT_BUCKET",
    "DEFAULT_LOCAL_DIR",
    "AssetStorage",
    "Backend",
    "ObjectStorageLike",
    "StorageSettings",
    "UploadResult",
    "build_client",
    "content_type_for",
]
