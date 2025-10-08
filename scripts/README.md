# Dataset Download Scripts

This directory contains scripts for downloading datasets used in the LLMVAD project. All the scripts suppose your datasets are hosted on Kaggle.

## Files

- **`download_data.py`** - Python wrapper for downloading datasets
- **`download_dataset.ps1`** - PowerShell script that handles the actual download using aria2c or fallback
- **`datasets_config.yaml`** - Configuration file defining available datasets

## Usage

### Download XD-Violence Dataset

```bash
# From project root
python scripts/download_data.py xd-violence

# Force re-download
python scripts/download_data.py xd-violence --force
```

### Download Custom Dataset

```bash
python scripts/download_data.py custom --slug user/dataset-name --name DatasetName --zip dataset.zip
```

## Configuration

Edit `datasets_config.yaml` to add new datasets:

```yaml
datasets:
  my-dataset:
    slug: kaggle-user/dataset-slug
    name: Dataset_Folder_Name
    zip: dataset-file.zip
    description: Description of the dataset
```

## Requirements

- Python 3.9+
- PyYAML (`pyyaml` package)
- PowerShell
- aria2c (optional, but recommended for faster downloads)
- Kaggle API credentials in `.env` file at project root

## Environment Setup

Create a `.env` file in the project root with your Kaggle credentials:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```
