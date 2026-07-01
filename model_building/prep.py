import os
from pathlib import Path
import pandas as pd
try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

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
from sklearn.model_selection import train_test_split
import urllib3

# Configure the Hugging Face client session used by hf:// to bypass SSL verification.
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

    if HfApi is not None:
        raise ImportError(
            "Installed huggingface_hub does not expose a supported HTTP client "
            "configuration hook."
        )

repo_id = "nikhileshmehta1989/Predictive_Maintenance_Vehicle"
LOCAL_DATA_PATH = Path("predictive_maintenance_vehicle/data/engine_data.csv")
HF_DATASET_PATH = f"hf://datasets/{repo_id}/engine_data.csv"

configure_huggingface_insecure_http()

if LOCAL_DATA_PATH.exists():
    print("Reading from local file:", LOCAL_DATA_PATH)
    df = pd.read_csv(LOCAL_DATA_PATH)
else:
    if HfApi is None:
        raise RuntimeError(
            f"Local source data is missing at {LOCAL_DATA_PATH} and "
            "huggingface_hub is not installed."
        )

    print("Reading from:", HF_DATASET_PATH)
    df = pd.read_csv(HF_DATASET_PATH)

print("Dataset loaded successfully.")

# Normalize column names from source file for stable processing.
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

target_col = "engine_condition"
if target_col not in df.columns:
    raise ValueError(f"Expected target column '{target_col}' not found. Columns: {list(df.columns)}")

df.drop(columns=["unnamed:_0", "customerid"], inplace=True, errors="ignore")

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])

print("Missing values handled.")

for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

if cat_cols:
    print("Categorical columns encoded.")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df[target_col] = y_train.values

test_df = X_test.copy()
test_df[target_col] = y_test.values

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

os.makedirs("predictive_maintenance_vehicle/data", exist_ok=True)
train_path = "predictive_maintenance_vehicle/data/train.csv"
test_path = "predictive_maintenance_vehicle/data/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print("Train and test datasets saved locally.")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN and HfApi is not None:
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=repo_id,
        repo_type="dataset",
    )

    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("Train and test datasets uploaded to Hugging Face successfully.")
elif HF_TOKEN and HfApi is None:
    print("HF_TOKEN is set, but huggingface_hub is not installed; skipping Hugging Face upload.")
else:
    print("HF_TOKEN is not set; skipping Hugging Face upload. Local train/test files are ready.")
