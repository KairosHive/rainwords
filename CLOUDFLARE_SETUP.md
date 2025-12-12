# Cloudflare Embeddings Setup for Railway

## Why Cloudflare?
Using Cloudflare Workers AI embeddings instead of local models:
- **Saves ~500MB RAM** (critical for Railway's free tier)
- **Faster cold starts** (no model loading)
- **Better performance** (Cloudflare's infrastructure)

## Setup Instructions

### 1. Get Cloudflare Credentials

1. Sign up at [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Go to **Workers & Pages** → **Overview**
3. Copy your **Account ID** (visible in the right sidebar)
4. Create an API token:
   - Go to **My Profile** → **API Tokens** → **Create Token**
   - Use template "Edit Cloudflare Workers"
   - Copy the generated token

### 2. Set Environment Variables in Railway

In your Railway project:
1. Go to **Settings** → **Variables**
2. Add these variables:

```
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_API_TOKEN=your_api_token_here
GEMINI_API_KEY=your_gemini_key_here
```

### 3. Deploy

Push to GitHub - Railway will auto-deploy with Cloudflare embeddings!

## Model Used

The app uses `@cf/baai/bge-small-en-v1.5` (384 dimensions), which:
- **Matches the dimension** of the original `all-MiniLM-L6-v2` (384d)
- Works with existing FAISS index (no rebuild needed)
- Free tier: 10M tokens/month (plenty for most apps)

## Fallback Behavior

If Cloudflare credentials are not found, the app automatically falls back to SentenceTransformer (local mode). This means:
- **Production (Railway)**: Uses Cloudflare (low RAM)
- **Local Development**: Uses SentenceTransformer (no API keys needed)

## Cost Estimate

Cloudflare Workers AI Free Tier:
- **10 million tokens/month** free
- Each embedding request ≈ 10-50 tokens
- Enough for **~200,000-500,000 queries/month**

## Troubleshooting

### "Cloudflare API error"
- Check that your API token has the correct permissions
- Verify Account ID is correct
- Check token hasn't expired

### Still using local model
- Verify environment variables are set in Railway dashboard
- Check deployment logs for "Using Cloudflare Workers AI" message
- Restart the deployment after adding variables
