"""
Cross-platform dataset downloader for LLMVAD project.

This script automatically detects the operating system and uses the appropriate
download script:
- Windows: download_dataset.ps1 (PowerShell)
- Linux/macOS: download_dataset.sh (Bash)

Both scripts support Kaggle dataset downloads with progress indicators and
fallback mechanisms (aria2c preferred, curl/WebRequest as fallback).
"""

import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


def load_dataset_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "datasets_config.yaml"

    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("datasets", {})


def _validate_dataset_params(
    dataset_slug: str, dataset_name: str, zip_name: str
) -> None:
    # Kaggle slug format: username/dataset-name (alphanumeric, hyphens, underscores, single slash)
    if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$", dataset_slug):
        msg = f"Invalid dataset_slug format: {dataset_slug}"
        raise ValueError(msg)

    # Dataset name: alphanumeric, hyphens, underscores, periods
    if not re.match(r"^[a-zA-Z0-9_.-]+$", dataset_name):
        msg = f"Invalid dataset_name format: {dataset_name}"
        raise ValueError(msg)

    # Zip name: alphanumeric, hyphens, underscores, must end with .zip
    if not re.match(r"^[a-zA-Z0-9_-]+\.zip$", zip_name):
        msg = f"Invalid zip_name format: {zip_name}"
        raise ValueError(msg)


def download_dataset(
    dataset_slug: str,
    dataset_name: str,
    zip_name: str,
    force: bool = False,
    script_path: Optional[Path] = None,
) -> int:
    _validate_dataset_params(dataset_slug, dataset_name, zip_name)

    # Determine which script to use based on the operating system
    if script_path is None:
        system = platform.system().lower()
        if system == "windows":
            script_path = Path(__file__).parent / "download_dataset.ps1"
        else:  # Linux, macOS, and other Unix-like systems
            script_path = Path(__file__).parent / "download_dataset.sh"

    if not script_path.exists():
        script_name = script_path.name
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        return 1

    try:
        script_path = script_path.resolve()
    except (OSError, RuntimeError) as e:
        print(f"Error resolving script path: {e}", file=sys.stderr)
        return 1

    # Build command based on script type
    system = platform.system().lower()
    if system == "windows" and script_path.suffix == ".ps1":
        cmd = [
            "powershell.exe",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-DatasetSlug",
            dataset_slug,
            "-DatasetName",
            dataset_name,
            "-ZipName",
            zip_name,
        ]
        if force:
            cmd.append("-Force")
    else:  # Unix-like systems with bash script
        cmd = [
            str(script_path),
            "--dataset-slug",
            dataset_slug,
            "--dataset-name",
            dataset_name,
            "--zip-name",
            zip_name,
        ]
        if force:
            cmd.append("--force")

    print(f"Downloading dataset: {dataset_slug}")
    print(f"Target folder: datasets/{dataset_name}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)  # noqa: S603
        return result.returncode
    except (subprocess.SubprocessError, OSError) as e:
        print(f"Error running script: {e}", file=sys.stderr)
        return 1


def download_xd_violence(force: bool = False) -> int:
    """Download the XD-Violence dataset."""
    datasets = load_dataset_config()
    if "xd-violence" not in datasets:
        print("Error: xd-violence dataset not found in config", file=sys.stderr)
        return 1

    config = datasets["xd-violence"]
    return download_dataset(
        dataset_slug=config["slug"],
        dataset_name=config["name"],
        zip_name=config["zip"],
        force=force,
    )


def main():
    """Main entry point for CLI usage."""
    import argparse

    datasets = {}
    try:
        datasets = load_dataset_config()
        available_datasets = list(datasets.keys())
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading dataset config: {e}", file=sys.stderr)
        available_datasets = []

    parser = argparse.ArgumentParser(
        description="Download datasets for LLMVAD project (supports Windows and Linux)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download XD-Violence dataset
  python download_data.py xd-violence

  # Force re-download
  python download_data.py xd-violence --force

  # Custom dataset
  python download_data.py custom --slug user/dataset --name MyDataset --zip dataset.zip

Note:
  - On Windows: Uses download_dataset.ps1 (PowerShell script)
  - On Linux/macOS: Uses download_dataset.sh (Bash script)
        """,
    )

    parser.add_argument(
        "dataset",
        choices=available_datasets + ["custom"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if dataset exists"
    )

    # Custom dataset options
    parser.add_argument("--slug", help="Kaggle dataset slug (for custom dataset)")
    parser.add_argument("--name", help="Dataset folder name (for custom dataset)")
    parser.add_argument("--zip", help="Zip file name (for custom dataset)")

    args = parser.parse_args()

    # Download dataset
    if args.dataset == "custom":
        if not all([args.slug, args.name, args.zip]):
            parser.error("--slug, --name, and --zip are required for custom dataset")
        exit_code = download_dataset(
            dataset_slug=args.slug,
            dataset_name=args.name,
            zip_name=args.zip,
            force=args.force,
        )
    elif args.dataset in available_datasets:
        config = datasets[args.dataset]
        exit_code = download_dataset(
            dataset_slug=config["slug"],
            dataset_name=config["name"],
            zip_name=config["zip"],
            force=args.force,
        )
    else:
        parser.error(f"Unknown dataset: {args.dataset}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
