from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()

# Create repo (one time)
api.create_repo(
    repo_id="Tiah90/voice-detection-data",
    repo_type="dataset",
    exist_ok=True
)

# Upload files
api.upload_folder(
    folder_path="preprocessed_features",
    repo_id="Tiah90/voice-detection-data",
    repo_type="dataset"
)

print("✓ Uploaded to Hugging Face")