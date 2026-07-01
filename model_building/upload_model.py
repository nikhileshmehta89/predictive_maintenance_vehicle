import os
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise ImportError(
        "huggingface_hub is required for Hugging Face upload. "
        "Install it with: pip install huggingface_hub"
    ) from exc

try:
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    from huggingface_hub.utils import RepositoryNotFoundError


MODEL_PATH = Path("predictive_maintenance_vehicle/model_building/best_decision_tree_model.pkl")
HF_SPACE_ID = (
    os.getenv("HF_SPACE_ID")
    or os.getenv("HF_UPLOAD_REPO_ID")
    or "nikhileshmehta1989/Predictive_Maintenance_Vehicle"
)
HF_UPLOAD_PATH_IN_REPO = os.getenv("HF_UPLOAD_PATH_IN_REPO", "best_model.pkl")
HF_SPACE_SDK = os.getenv("HF_SPACE_SDK", "gradio")
HF_TOKEN = os.getenv("HF_TOKEN")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Run train.py first.")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set; cannot upload model to Hugging Face.")

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=HF_SPACE_ID, repo_type="space")
    print(f"Hugging Face Space '{HF_SPACE_ID}' already exists.")
except RepositoryNotFoundError:
    api.create_repo(
        repo_id=HF_SPACE_ID,
        repo_type="space",
        private=False,
        exist_ok=True,
        space_sdk=HF_SPACE_SDK,
    )
    print(f"Hugging Face Space '{HF_SPACE_ID}' created.")

commit_info = api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo=HF_UPLOAD_PATH_IN_REPO,
    repo_id=HF_SPACE_ID,
    repo_type="space",
    commit_message="Upload tuned DecisionTree model",
)

print(f"Uploaded model to Hugging Face Space at {HF_SPACE_ID}/{HF_UPLOAD_PATH_IN_REPO}.")
if getattr(commit_info, "commit_url", None):
    print(f"Commit URL: {commit_info.commit_url}")
