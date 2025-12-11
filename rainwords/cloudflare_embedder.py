"""
Cloudflare Workers AI Embeddings wrapper for Rainwords.
Uses Cloudflare's API instead of loading models locally to save RAM.
"""
import os
import numpy as np
import requests
import time
from typing import List, Optional


class CloudflareEmbedder:
    """
    Embedding model that uses Cloudflare Workers AI API.
    Zero local RAM usage for models.
    """
    
    def __init__(self, model: str = "@cf/baai/bge-large-en-v1.5"):
        """
        Initialize Cloudflare embedder.
        
        Args:
            model: Cloudflare model ID. Options:
                - "@cf/baai/bge-small-en-v1.5" (384 dim)
                - "@cf/baai/bge-base-en-v1.5" (768 dim)
                - "@cf/baai/bge-large-en-v1.5" (1024 dim)
        """
        self.model = model
        self.account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self.api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        
        if not self.account_id or not self.api_token:
            raise ValueError(
                "Cloudflare credentials not found. Set CLOUDFLARE_ACCOUNT_ID "
                "and CLOUDFLARE_API_TOKEN environment variables."
            )
        
        # Dimension mapping
        self.dim_map = {
            "@cf/baai/bge-small-en-v1.5": 384,
            "@cf/baai/bge-base-en-v1.5": 768,
            "@cf/baai/bge-large-en-v1.5": 1024,
        }
        self.dimension = self.dim_map.get(model, 768)
        
        print(f"[CloudflareEmbedder] Initialized: {model} (dim={self.dimension})")
    
    def encode(self, texts: List[str], convert_to_numpy: bool = True, 
               convert_to_tensor: bool = False, batch_size: int = 50, 
               max_retries: int = 3) -> np.ndarray:
        """
        Encode texts to embeddings using Cloudflare API.
        
        Args:
            texts: List of strings to embed
            convert_to_numpy: Return numpy array (for compatibility)
            convert_to_tensor: If True, return torch tensor instead of numpy
            batch_size: Max texts per API call
            max_retries: Retry attempts on failure
            
        Returns:
            numpy array or torch tensor of shape (len(texts), dimension)
        """
        if not texts:
            if convert_to_tensor:
                import torch
                return torch.tensor([], dtype=torch.float32)
            return np.array([], dtype=np.float32)
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._call_api(batch, max_retries)
            all_embeddings.append(embeddings)
        
        result = np.concatenate(all_embeddings, axis=0)
        
        # Normalize embeddings (important for cosine similarity)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / (norms + 1e-9)
        
        result = result.astype(np.float32)
        
        # Convert to tensor if requested
        if convert_to_tensor:
            import torch
            return torch.from_numpy(result)
        
        return result
    
    def _call_api(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        """
        Call Cloudflare embedding API with retry logic.
        """
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json={"text": texts},
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    raise RuntimeError(f"Cloudflare API error: {data.get('errors', [])}")
                
                # Extract embeddings from response
                result = data.get("result", {})
                
                if isinstance(result, dict) and "data" in result:
                    embeddings = result["data"]
                elif isinstance(result, list):
                    embeddings = result
                else:
                    raise RuntimeError(f"Unexpected response format: {type(result)}")
                
                return np.array(embeddings, dtype=np.float32)
                
            except requests.exceptions.HTTPError as e:
                last_error = e
                if response.status_code == 500 and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                    print(f"[CloudflareEmbedder] 500 error, retrying in {wait_time}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise
        
        raise last_error
    
    def get_sentence_embedding_dimension(self) -> int:
        """Compatibility with SentenceTransformer API."""
        return self.dimension


def create_embedder(model_name: str = 'all-MiniLM-L6-v2') -> object:
    """
    Factory function to create embedder based on available credentials.
    
    Returns Cloudflare embedder if credentials exist, otherwise SentenceTransformer.
    """
    # Try Cloudflare first
    cf_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    cf_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    
    if cf_account and cf_token:
        print("[Embedder] Using Cloudflare Workers AI (zero local RAM)")
        # Use BGE-large which is 1024 dim for better quality
        cf_model = "@cf/baai/bge-large-en-v1.5"  # 1024 dim
        
        return CloudflareEmbedder(model=cf_model)
    
    # Fallback to SentenceTransformer
    print(f"[Embedder] Using SentenceTransformer: {model_name} (local)")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
