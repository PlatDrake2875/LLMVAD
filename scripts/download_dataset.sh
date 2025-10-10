#!/bin/bash
# Kaggle dataset downloader for Linux (aria2 with live progress; curl fallback with progress bar)
# - Loads KAGGLE_USERNAME/KAGGLE_KEY from .env file in project root
# - Uses aria2c if available for faster multi-connection downloads
# - Falls back to curl with progress bar if aria2c not found
# - Extracts dataset to datasets/<DatasetName>/ folder

set -euo pipefail

# Default values
FORCE=false
DATASET_SLUG="adrianpatrascu/xd-violence-1-1004"
DATASET_NAME="XD_Violence_1-1004"
ZIP_NAME="xd-violence.zip"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --dataset-slug)
            DATASET_SLUG="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --zip-name)
            ZIP_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --force              Force re-download even if dataset exists"
            echo "  --dataset-slug SLUG  Kaggle dataset slug (default: $DATASET_SLUG)"
            echo "  --dataset-name NAME  Dataset folder name (default: $DATASET_NAME)"
            echo "  --zip-name ZIP       Zip file name (default: $ZIP_NAME)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASETS_DIR="$PROJECT_ROOT/datasets"
DATASET_PATH="$DATASETS_DIR/$DATASET_NAME"
ZIP_PATH="$DATASETS_DIR/$ZIP_NAME"
DOT_ENV_PATH="$PROJECT_ROOT/.env"

# Kaggle dataset URL
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/$DATASET_SLUG"

echo "Kaggle Dataset Downloader (aria2 + progress)"

# --- Load .env file ---
load_dotenv() {
    local env_file="$1"
    if [[ ! -f "$env_file" ]]; then
        return
    fi
    
    # Read .env file and export variables
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ "$line" =~ ^[[:space:]]*$ ]]; then
            continue
        fi
        
        # Remove 'export ' prefix if present
        line="${line#export }"
        
        # Extract key=value pairs
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"
            
            # Remove inline comments (not inside quotes)
            if [[ ! "$value" =~ ^[\"\'] ]]; then
                value="${value%%[[:space:]]*#*}"
                value="${value%"${value##*[![:space:]]}"}" # trim trailing whitespace
            fi
            
            # Remove quotes if present
            if [[ "$value" =~ ^\"(.*)\"$ ]] || [[ "$value" =~ ^\'(.*)\'$ ]]; then
                value="${BASH_REMATCH[1]}"
            fi
            
            export "$key"="$value"
        fi
    done < "$env_file"
}

load_dotenv "$DOT_ENV_PATH"

# --- Get Kaggle credentials ---
get_kaggle_credentials() {
    if [[ -z "${KAGGLE_USERNAME:-}" ]] || [[ -z "${KAGGLE_KEY:-}" ]]; then
        echo "Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env file" >&2
        exit 1
    fi
    echo "$KAGGLE_USERNAME:$KAGGLE_KEY"
}

# --- Skip if already present ---
if [[ -d "$DATASET_PATH" ]] && [[ "$FORCE" != "true" ]]; then
    if [[ -n "$(ls -A "$DATASET_PATH" 2>/dev/null)" ]]; then
        item_count=$(find "$DATASET_PATH" -mindepth 1 | wc -l)
        echo "Dataset already exists with $item_count items. Use --force to re-download."
        exit 0
    fi
fi

# --- Ensure folder & cleanup ---
mkdir -p "$DATASETS_DIR"
[[ -f "$ZIP_PATH" ]] && rm -f "$ZIP_PATH"
[[ -d "$DATASET_PATH" ]] && rm -rf "$DATASET_PATH"

# Get credentials
CREDENTIALS=$(get_kaggle_credentials)
AUTH_HEADER="Authorization: Basic $(echo -n "$CREDENTIALS" | base64 -w 0)"

# --- Try aria2c with live progress ---
if command -v aria2c >/dev/null 2>&1; then
    echo "Downloading via Kaggle API (aria2c)..."
    aria2c \
        --header="$AUTH_HEADER" \
        --header="Accept: application/zip" \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=1M \
        --continue=true \
        --allow-overwrite=true \
        --auto-file-renaming=false \
        --file-allocation=none \
        --summary-interval=0 \
        --console-log-level=warn \
        --show-console-readout=true \
        --dir="$DATASETS_DIR" \
        --out="$ZIP_NAME" \
        "$DATASET_URL"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: aria2c failed" >&2
        exit 1
    fi
else
    # --- Fallback: curl with progress bar ---
    echo "aria2c not found. Falling back to curl with progress..."
    curl \
        --location \
        --header "$AUTH_HEADER" \
        --header "Accept: application/zip" \
        --progress-bar \
        --output "$ZIP_PATH" \
        "$DATASET_URL"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: curl failed" >&2
        exit 1
    fi
fi

if [[ ! -f "$ZIP_PATH" ]]; then
    echo "Error: Download failed - file not created" >&2
    exit 1
fi

# --- Sanity check: ZIP signature "PK" ---
if ! file "$ZIP_PATH" | grep -q "Zip archive"; then
    echo "Error: Downloaded file is not a valid ZIP archive" >&2
    echo "File type: $(file "$ZIP_PATH")"
    head -c 200 "$ZIP_PATH" | hexdump -C
    exit 1
fi

# --- Extract ---
echo "Extracting..."
mkdir -p "$DATASET_PATH"

if command -v unzip >/dev/null 2>&1; then
    unzip -q "$ZIP_PATH" -d "$DATASET_PATH"
elif command -v 7z >/dev/null 2>&1; then
    7z x "$ZIP_PATH" -o"$DATASET_PATH" >/dev/null
else
    echo "Error: No extraction tool found (unzip or 7z required)" >&2
    exit 1
fi

rm -f "$ZIP_PATH"

# Check extraction success
if [[ -n "$(ls -A "$DATASET_PATH" 2>/dev/null)" ]]; then
    item_count=$(find "$DATASET_PATH" -mindepth 1 | wc -l)
    echo "Extraction completed."
    echo "Dataset ready at: $DATASET_PATH"
    echo "Items: $item_count"
else
    echo "Error: Extraction failed - no files found" >&2
    exit 1
fi