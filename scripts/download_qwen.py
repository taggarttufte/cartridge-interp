"""Pre-download Qwen3-4B weights to the HF cache."""

from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="Qwen/Qwen3-4B")
print("Qwen3-4B downloaded to:", path)
