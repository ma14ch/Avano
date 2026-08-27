#!/bin/bash

# Downloads Whisper models (defined in WHISPER_MODELS, src/models.py) into the
# shared Hugging Face cache (./hf-cache) used by docker-compose.yml. This is
# the same cache deploy.sh reads from when picking which model to run.
#
# Usage:
#   ./download.sh                # interactive menu
#   ./download.sh <model-key>    # download one model, e.g. ./download.sh persian-v4
#   ./download.sh all            # download every model in WHISPER_MODELS

set -e

PROJECT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$PROJECT_DIR"

HF_CACHE_DIR="$PROJECT_DIR/hf-cache/transformers"

# Parse the WHISPER_MODELS dict straight out of src/models.py so this script
# never goes out of sync with the models actually supported by the app.
parse_whisper_models() {
    sed -n '/^WHISPER_MODELS = {/,/^}/p' "$PROJECT_DIR/src/models.py" \
        | grep -E '^\s*"[^"]+"\s*:\s*"[^"]+"' \
        | sed -E 's/^[[:space:]]*"([^"]+)"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1 \2/'
}

model_cache_dir_name() {
    echo "models--${1//\//--}"
}

is_model_downloaded() {
    local snapshots_dir="$HF_CACHE_DIR/$(model_cache_dir_name "$1")/snapshots"
    [ -d "$snapshots_dir" ] && [ -n "$(ls -A "$snapshots_dir" 2>/dev/null)" ]
}

mapfile -t MODEL_ENTRIES < <(parse_whisper_models)
MODEL_KEYS=()
MODEL_REPOS=()
for entry in "${MODEL_ENTRIES[@]}"; do
    MODEL_KEYS+=("${entry%% *}")
    MODEL_REPOS+=("${entry#* }")
done

if [ "${#MODEL_KEYS[@]}" -eq 0 ]; then
    echo "Error: could not find any WHISPER_MODELS entries in src/models.py"
    exit 1
fi

print_model_list() {
    echo "Available Whisper models:"
    for i in "${!MODEL_KEYS[@]}"; do
        local status="not downloaded"
        is_model_downloaded "${MODEL_REPOS[$i]}" && status="downloaded"
        printf "  %d) %-14s %-45s [%s]\n" "$((i+1))" "${MODEL_KEYS[$i]}" "${MODEL_REPOS[$i]}" "$status"
    done
}

download_model_key() {
    local key="$1"
    local repo=""
    for i in "${!MODEL_KEYS[@]}"; do
        if [ "${MODEL_KEYS[$i]}" = "$key" ]; then
            repo="${MODEL_REPOS[$i]}"
            break
        fi
    done
    if [ -z "$repo" ]; then
        echo "Error: unknown model key '$key'"
        print_model_list
        exit 1
    fi
    echo "Downloading '$key' ($repo) into $HF_CACHE_DIR ..."
    docker compose run --rm ai-tts python3 -c "from src.models import download_whisper_model; download_whisper_model('$repo')"
}

download_all_models() {
    echo "Downloading all Whisper models into $HF_CACHE_DIR ..."
    docker compose run --rm ai-tts python3 -c "from src.models import download_all_whisper_models; download_all_whisper_models()"
}

echo "Building image (if needed) so models can be downloaded through the container..."
docker compose build ai-tts

case "${1:-}" in
    "")
        print_model_list
        echo
        read -rp "Select a model to download [1-${#MODEL_KEYS[@]}] or 'a' for all: " choice
        if [[ "$choice" =~ ^[Aa]$ ]]; then
            download_all_models
        elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#MODEL_KEYS[@]}" ]; then
            download_model_key "${MODEL_KEYS[$((choice-1))]}"
        else
            echo "Invalid selection."
            exit 1
        fi
        ;;
    all)
        download_all_models
        ;;
    *)
        download_model_key "$1"
        ;;
esac

echo
echo "Done. Current cache status:"
print_model_list

