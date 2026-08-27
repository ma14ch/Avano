#!/bin/bash

# Downloads ASR models (defined in ASR_MODEL_REPOS, src/models.py) into the
# shared Hugging Face cache (./hf-cache) used by docker-compose.yml. This is
# the same cache deploy.sh reads from when picking which model to run.
#
# Models can belong to different families (e.g. Whisper encoder-decoder
# checkpoints, or Qwen3-ASR Transformers-native models) - the download itself
# is dispatched through src/models.py's download_asr_model(), which knows how
# to fetch the right files for each family.
#
# Usage:
#   ./download.sh                # interactive menu
#   ./download.sh <model-key>    # download one model, e.g. ./download.sh persian-v4
#   ./download.sh all            # download every model in ASR_MODEL_REPOS

set -e

PROJECT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$PROJECT_DIR"

HF_CACHE_DIR="$PROJECT_DIR/hf-cache/transformers"

# Parse the ASR_MODEL_REPOS / ASR_MODEL_FAMILIES dicts straight out of
# src/models.py so this script never goes out of sync with the models
# actually supported by the app.
parse_dict() {
    sed -n "/^$1 = {/,/^}/p" "$PROJECT_DIR/src/models.py" \
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

mapfile -t MODEL_ENTRIES < <(parse_dict ASR_MODEL_REPOS)
mapfile -t FAMILY_ENTRIES < <(parse_dict ASR_MODEL_FAMILIES)
MODEL_KEYS=()
MODEL_REPOS=()
for entry in "${MODEL_ENTRIES[@]}"; do
    MODEL_KEYS+=("${entry%% *}")
    MODEL_REPOS+=("${entry#* }")
done
declare -A MODEL_FAMILY
for entry in "${FAMILY_ENTRIES[@]}"; do
    MODEL_FAMILY["${entry%% *}"]="${entry#* }"
done

if [ "${#MODEL_KEYS[@]}" -eq 0 ]; then
    echo "Error: could not find any ASR_MODEL_REPOS entries in src/models.py"
    exit 1
fi

print_model_list() {
    echo "Available ASR models:"
    for i in "${!MODEL_KEYS[@]}"; do
        local key="${MODEL_KEYS[$i]}"
        local status="not downloaded"
        is_model_downloaded "${MODEL_REPOS[$i]}" && status="downloaded"
        printf "  %d) %-16s %-45s family=%-10s [%s]\n" "$((i+1))" "$key" "${MODEL_REPOS[$i]}" "${MODEL_FAMILY[$key]:-whisper}" "$status"
    done
}

download_model_key() {
    local key="$1"
    local found=false
    for k in "${MODEL_KEYS[@]}"; do
        if [ "$k" = "$key" ]; then
            found=true
            break
        fi
    done
    if [ "$found" != true ]; then
        echo "Error: unknown model key '$key'"
        print_model_list
        exit 1
    fi
    echo "Downloading '$key' into $HF_CACHE_DIR ..."
    docker compose run --rm ai-tts python3 -c "from src.models import download_asr_model; download_asr_model('$key')"
}

download_all_models() {
    echo "Downloading all ASR models into $HF_CACHE_DIR ..."
    docker compose run --rm ai-tts python3 -c "from src.models import download_all_asr_models; download_all_asr_models()"
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

