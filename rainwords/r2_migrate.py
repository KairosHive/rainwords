"""
One-off helper: verify Cloudflare R2 connectivity and migrate any locally-stored
uploaded-corpus shards (./user_data) into the R2 bucket.

Usage (after filling R2_* in rainwords/.env):
    uv run python -m rainwords.r2_migrate            # verify + migrate
    uv run python -m rainwords.r2_migrate --check     # verify only

Safe to re-run: objects already present in R2 are skipped.
"""
import sys
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from .storage import LocalStorage, create_storage, LOCAL_ROOT  # noqa: E402


def main() -> None:
    check_only = "--check" in sys.argv

    # 1) Connectivity check — create_storage() must resolve to R2.
    remote = create_storage()
    if type(remote).__name__ != "R2Storage":
        print("✗ R2 is NOT configured — create_storage() returned "
              f"{type(remote).__name__}.")
        print("  Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET "
              "(and CLOUDFLARE_ACCOUNT_ID) in rainwords/.env, then retry.")
        sys.exit(1)

    probe = "._healthcheck"
    try:
        remote.put_bytes(probe, b"ok")
        assert remote.get_bytes(probe) == b"ok"
        remote.delete_prefix(probe)
    except Exception as e:
        print(f"✗ R2 connectivity failed: {e}")
        sys.exit(1)
    print(f"✓ R2 reachable — bucket '{os.environ.get('R2_BUCKET')}'")

    if check_only:
        return

    # 2) Migrate local shards -> R2 (skip anything already there).
    local = LocalStorage()
    keys = local.list_prefix("owners/")
    if not keys:
        print(f"(no local shards under {LOCAL_ROOT / 'owners'} — nothing to migrate)")
        return

    copied = skipped = 0
    for k in keys:
        if remote.exists(k):
            skipped += 1
            continue
        data = local.get_bytes(k)
        if data is None:
            continue
        remote.put_bytes(k, data)
        copied += 1
        print(f"  → {k}")
    print(f"✓ Migration done: {copied} uploaded, {skipped} already present "
          f"({len(keys)} local objects total).")


if __name__ == "__main__":
    main()
