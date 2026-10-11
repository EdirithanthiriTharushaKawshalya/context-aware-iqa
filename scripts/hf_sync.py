"""
Hugging Face Dataset Synchronization Utility
Allows uploading raw event photographs to Hugging Face Datasets
and downloading them onto any local workstation or server.
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from huggingface_hub import HfApi, snapshot_download, login
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("[ERROR] 'huggingface_hub' is required. Run: pip install huggingface_hub")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync dataset between local storage and Hugging Face Datasets Hub."
    )
    parser.add_argument(
        "action",
        choices=["upload", "download", "status"],
        help="Action to perform: 'upload', 'download', or 'status'.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face Dataset repo ID (format: 'username/dataset-name').",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Local directory to upload from or download to (default: data/raw).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face API token. Defaults to $HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If creating a new dataset repo on upload, create it as Private.",
    )
    return parser.parse_args()


def upload_dataset(repo_id: str, local_dir: Path, token: str, private: bool):
    """Uploads the local dataset directory to Hugging Face Datasets."""
    if not local_dir.is_dir():
        print(f"[ERROR] Local directory does not exist: {local_dir}")
        sys.exit(1)

    print(f"\n[INFO] Initializing Hugging Face Upload...")
    print(f"Target Repo:  https://huggingface.co/datasets/{repo_id}")
    print(f"Source Dir:   {local_dir.resolve()}")
    print(f"Private Repo: {private}")

    api = HfApi(token=token)

    # 1. Create dataset repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"[SUCCESS] Repository '{repo_id}' ready on Hugging Face.")
    except Exception as e:
        print(f"[WARNING] Could not verify/create repo '{repo_id}': {e}")

    # 2. Upload folder
    print(f"\n[INFO] Uploading files to Hugging Face Hub (this may take a moment for large images)...")
    try:
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Sync dataset photos via Context-Aware IQA pipeline",
        )
        print(f"\n[SUCCESS] Dataset successfully uploaded to:")
        print(f"URL: https://huggingface.co/datasets/{repo_id}\n")
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")
        sys.exit(1)


def download_dataset(repo_id: str, local_dir: Path, token: str):
    """Downloads dataset from Hugging Face into local data/raw/."""
    print(f"\n[INFO] Downloading dataset from Hugging Face...")
    print(f"Source Repo:  https://huggingface.co/datasets/{repo_id}")
    print(f"Target Dir:   {local_dir.resolve()}")

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=token,
        )
        print(f"\n[SUCCESS] Dataset downloaded successfully into: {downloaded_path}\n")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)

    if args.action == "upload":
        if not args.token:
            print("[ERROR] HF token is required for upload. Provide --token or set $HF_TOKEN.")
            sys.exit(1)
        upload_dataset(args.repo_id, local_dir, args.token, args.private)

    elif args.action == "download":
        download_dataset(args.repo_id, local_dir, args.token)

    elif args.action == "status":
        api = HfApi(token=args.token)
        try:
            info = api.dataset_info(repo_id=args.repo_id)
            print(f"\n[INFO] Hugging Face Dataset: {info.id}")
            print(f"Author:     {info.author}")
            print(f"Private:    {info.private}")
            print(f"Files:      {len(info.siblings or [])} tracked objects\n")
        except Exception as e:
            print(f"[ERROR] Could not fetch dataset info: {e}")


if __name__ == "__main__":
    main()
