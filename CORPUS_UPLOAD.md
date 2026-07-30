# Uploading Corpora (claimed handle, no sign-in)

Users can add their own corpus (PDF or `.txt`) straight from the web UI. Open the
**Text Sources** dropdown → *Your corpora* → pick a **handle** → **Upload PDF / TXT**.

## How it works

1. **Identity — a claimed handle, not an account.** The handle you type is saved
   in the browser's `localStorage` and sent as an `X-Rainwords-Handle` header on
   every request. There is no password and nothing is stored server-side about
   *you* — the handle is just a namespace key.
2. **Sanitize.** PDFs go through `text_pipeline.clean_text` (de-hyphenation,
   page-number stripping, paragraph rebuild); `.txt` files are glyph-normalized
   but keep their blank-line stanza structure.
3. **Chunk + embed.** The text is split into stanzas (`chunk_text`) and embedded
   with the *same* model as the built-in index, so uploads are searchable
   alongside the shipped corpora.
4. **Persist as a per-owner shard** under `owners/<handle>/<corpus_id>/`
   (`vectors.npy`, `docs.json`, `source.txt`, `meta.json`). The corpus is added to
   the in-memory index immediately, so it's usable without a restart.
5. **Reconnect.** Returning with the same handle lazy-loads that handle's shards
   from storage, so the corpora reappear. Other handles' uploads are never listed
   or searched for you.

> **Privacy note:** the handle is a bearer token. It's unlisted, not
> authenticated — anyone who types your handle can see and add to its corpora,
> and clearing browser storage loses the handle (the data still exists, but you
> can't prove ownership). Fine for poetry; don't upload anything sensitive.

## Where uploads are stored (persistence)

Storage is chosen automatically at startup (mirrors the embedder fallback):

| Environment | Backend | Persists across Railway redeploys? |
|---|---|---|
| `R2_*` vars set | Cloudflare R2 (S3 API) | ✅ yes |
| `RAINWORDS_DATA_DIR` set to a mounted disk | that folder | ✅ yes (e.g. a Railway Volume at `/data`) |
| neither | local `user_data/` folder | ❌ no — lost on redeploy (fine for local/desktop use) |

**Important:** the default container filesystem on Railway is ephemeral. For the
deployed app you must pick one of the two persistent options above.

### Option A — Cloudflare R2 (recommended, no sign-in)

Create an R2 bucket, then set (server-side infra secrets, same category as your
Workers AI token — not user credentials):

```
R2_ACCOUNT_ID=your_cloudflare_account_id     # falls back to CLOUDFLARE_ACCOUNT_ID
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret
R2_BUCKET=your_bucket_name
# optional: R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
```

### Option B — Railway Volume

Attach a Volume (e.g. mounted at `/data`) and set `RAINWORDS_DATA_DIR=/data`.
No R2 needed; uploads persist on the volume.

## Notes / limits

- **Embedding model must match the built-in index.** In production the server uses
  Cloudflare BGE-large (1024-dim), which is what `poetry.index` was built with. A
  shard embedded at a different dimension is skipped at load time (logged).
- No OCR: scanned/image-only PDFs (no text layer) are rejected with a clear error.
- Caps: 20 MB per file, 40 corpora per handle.
- Rarity filter: the global word-frequency map covers built-in corpora only, so
  words unique to an upload are treated as "rare".
