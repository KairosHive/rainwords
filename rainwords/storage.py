"""
Durable blob storage for user-uploaded corpora.

Mirrors the `create_embedder` fallback pattern: if Cloudflare R2 credentials
are present in the environment, uploads persist to R2 (survives Railway
redeploys). Otherwise they persist to a local `user_data/` folder, which is
perfect for local development (persists on the developer's disk, no creds).

Nothing here involves user sign-in: R2 credentials are server-side
infrastructure secrets, in the same category as CLOUDFLARE_API_TOKEN.
"""
import os
from pathlib import Path
from typing import List, Optional

# Local fallback root = <repo>/user_data  (repo root is the parent of the package)
BASE_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = Path(os.environ.get("RAINWORDS_DATA_DIR", BASE_DIR.parent / "user_data"))


class Storage:
    """Minimal key/value blob interface (keys look like 'owners/foo/bar/meta.json')."""

    def put_bytes(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def get_bytes(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def list_prefix(self, prefix: str) -> List[str]:
        raise NotImplementedError

    def delete_prefix(self, prefix: str) -> None:
        raise NotImplementedError


class LocalStorage(Storage):
    """Stores blobs as files under LOCAL_ROOT. Keys map 1:1 to relative paths."""

    def __init__(self, root: Path = LOCAL_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        print(f"[Storage] Using local disk: {self.root}")

    def _path(self, key: str) -> Path:
        # Keys are POSIX-style; forbid escaping the root.
        rel = Path(key.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe storage key: {key!r}")
        return self.root / rel

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        if not p.exists():
            return None
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_prefix(self, prefix: str) -> List[str]:
        base = self._path(prefix)
        # `prefix` may point at a directory or a partial name; walk from its dir.
        search_dir = base if base.is_dir() else base.parent
        if not search_dir.exists():
            return []
        keys: List[str] = []
        for f in search_dir.rglob("*"):
            if f.is_file():
                key = f.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def delete_prefix(self, prefix: str) -> None:
        for key in self.list_prefix(prefix):
            self._path(key).unlink(missing_ok=True)


class R2Storage(Storage):
    """Stores blobs in a Cloudflare R2 bucket via the S3-compatible API (boto3)."""

    def __init__(self, account_id: str, access_key: str, secret_key: str, bucket: str):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        endpoint = os.environ.get(
            "R2_ENDPOINT_URL",
            f"https://{account_id}.r2.cloudflarestorage.com",
        )
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )
        print(f"[Storage] Using Cloudflare R2: bucket={bucket}")

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> Optional[bytes]:
        from botocore.exceptions import ClientError
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def list_prefix(self, prefix: str) -> List[str]:
        keys: List[str] = []
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return sorted(keys)

    def delete_prefix(self, prefix: str) -> None:
        keys = self.list_prefix(prefix)
        for i in range(0, len(keys), 1000):
            batch = [{"Key": k} for k in keys[i:i + 1000]]
            if batch:
                self.client.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": batch}
                )


def create_storage() -> Storage:
    """
    Factory: R2 when credentials + bucket are set, else local disk.
    """
    account_id = os.environ.get("R2_ACCOUNT_ID") or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET")

    if account_id and access_key and secret_key and bucket:
        try:
            return R2Storage(account_id, access_key, secret_key, bucket)
        except Exception as e:
            print(f"[Storage] R2 init failed ({e}); falling back to local disk.")

    return LocalStorage()
