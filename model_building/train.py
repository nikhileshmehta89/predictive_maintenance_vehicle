import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import parallel_backend
try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None
    hf_hub_download = None

try:
    from huggingface_hub import set_client_factory
except ImportError:
    set_client_factory = None

try:
    from huggingface_hub import configure_http_backend
except ImportError:
    try:
        from huggingface_hub.utils import configure_http_backend
    except ImportError:
        configure_http_backend = None

try:
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    try:
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        RepositoryNotFoundError = Exception
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import urllib3

# Configure the Hugging Face client session to bypass SSL verification.
# This is needed in environments where corporate/root CA certs are not available.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def configure_huggingface_insecure_http() -> None:
    if configure_http_backend is not None:
        import requests

        def insecure_backend_factory() -> requests.Session:
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory=insecure_backend_factory)
        return

    if set_client_factory is not None:
        import httpx

        def insecure_client_factory() -> httpx.Client:
            return httpx.Client(verify=False)

        set_client_factory(insecure_client_factory)
        return

    if HfApi is not None or hf_hub_download is not None:
        raise ImportError(
            "Installed huggingface_hub does not expose a supported HTTP client "
            "configuration hook."
        )

configure_huggingface_insecure_http()

# Prefer Hugging Face splits, then fall back to local files if the download fails.
DATASET_REPO_ID = "nikhileshmehta1989/Predictive_Maintenance_Vehicle"
LOCAL_DATA_DIR = Path("data")
LOCAL_TRAIN_PATH = LOCAL_DATA_DIR / "train.csv"
LOCAL_TEST_PATH = LOCAL_DATA_DIR / "test.csv"
TARGET_COL = "engine_condition"

def read_split_csv(path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if df.empty and len(df.columns) == 1 and "temporary redirect" in str(df.columns[0]).lower():
        raise ValueError(
            f"{split_name} did not load as CSV data. The response was a Hugging Face "
            "redirect page, which usually means the remote read timed out."
        )

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COL}' in {split_name} data. "
            f"Columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"{split_name} data loaded successfully but has 0 rows.")

    return df

def load_split_csv(local_path: Path, filename: str) -> pd.DataFrame:
    if hf_hub_download is not None:
        try:
            print(f"Downloading {filename} from Hugging Face...")
            downloaded_path = hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=os.getenv("HF_TOKEN"),
            )
            return read_split_csv(downloaded_path, filename)
        except Exception as exc:
            if not local_path.exists():
                raise RuntimeError(
                    f"Could not load {filename} from Hugging Face and no local "
                    f"fallback exists at {local_path}."
                ) from exc
            print(f"Hugging Face download failed for {filename}; loading local fallback.")

    if not local_path.exists():
        raise RuntimeError(
            f"Could not load {filename}. Install huggingface_hub or run prep.py "
            "to create local train/test CSVs."
        )

    print(f"Loading {filename} from local file: {local_path}")
    return read_split_csv(local_path, filename)

train_df = load_split_csv(LOCAL_TRAIN_PATH, "train.csv")
test_df = load_split_csv(LOCAL_TEST_PATH, "test.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# MLflow experiment setup using local file-based tracking (no server required).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
else:
    # In CI, avoid committed local mlruns metadata that may contain absolute
    # artifact paths from a different machine (for example /Users/...).
    tracking_dir = Path(".mlruns_ci" if os.getenv("GITHUB_ACTIONS") == "true" else "mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow_tracking_uri = tracking_dir.as_uri()

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Predictive_Maintenance_Vehicle")
print(f"MLflow tracking URI set to: {mlflow_tracking_uri}")

# Define one model and tune hyperparameters for that model.
model_name = "DecisionTree"
estimator = DecisionTreeClassifier(random_state=42)
default_n_jobs = min(4, os.cpu_count() or 1)
n_jobs = int(os.getenv("SKLEARN_N_JOBS", str(default_n_jobs)))
grid_search_verbose = int(os.getenv("GRID_SEARCH_VERBOSE", "2"))
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 0.001, 0.01],
}

print(f"\nTuning {model_name}...")
print(f"Grid search n_jobs: {n_jobs} with threading backend")
print(f"Grid search verbose: {grid_search_verbose}")
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=n_jobs,
    verbose=grid_search_verbose,
)
with parallel_backend("threading"):
    grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
}

with mlflow.start_run(run_name=model_name):
    mlflow.log_param("model_name", model_name)
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(best_model, artifact_path="model")

print(f"Best params : {best_params}")
print(f"Best CV F1  : {grid_search.best_score_:.4f}")
print(f"Test F1     : {metrics['f1_score']:.4f}")
print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")

# Save tuned Decision Tree locally.
os.makedirs("model_building", exist_ok=True)
model_path = "model_building/best_decision_tree_model.pkl"
joblib.dump(best_model, model_path)
print(f"Tuned Decision Tree model saved to {model_path}")

cv_results_path = "model_building/grid_search_results.csv"
pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path, index=False)
print(f"Grid search results saved to {cv_results_path}")

# Upload tuned Decision Tree to the Hugging Face Space.
HF_SPACE_ID = (
    os.getenv("HF_SPACE_ID")
    or os.getenv("HF_UPLOAD_REPO_ID")
    or "nikhileshmehta1989/Predictive_Maintenance_Vehicle"
)
HF_UPLOAD_PATH_IN_REPO = os.getenv("HF_UPLOAD_PATH_IN_REPO", "best_model.pkl")
HF_SPACE_SDK = os.getenv("HF_SPACE_SDK", "gradio")
HF_TOKEN = os.getenv("HF_TOKEN")

if HfApi is None:
    raise ImportError(
        "huggingface_hub is required for Hugging Face upload. "
        "Install it with: pip install huggingface_hub"
    )

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
    path_or_fileobj=model_path,
    path_in_repo=HF_UPLOAD_PATH_IN_REPO,
    repo_id=HF_SPACE_ID,
    repo_type="space",
    commit_message=f"Upload tuned {model_name} model",
)
print(f"Tuned {model_name} uploaded to Hugging Face Space at {HF_SPACE_ID}/{HF_UPLOAD_PATH_IN_REPO}.")
if getattr(commit_info, "commit_url", None):
    print(f"Commit URL: {commit_info.commit_url}")
