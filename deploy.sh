
PROJECT_DIR="$(dirname "$(readlink -f "$0")")"

# Navigate to project directory
cd "$PROJECT_DIR"
echo "Deploying from directory: $PROJECT_DIR"

HF_CACHE_DIR="$PROJECT_DIR/hf-cache/transformers"

# --- ASR model discovery -----------------------------------------------------
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

repo_for_key() {
    local key="$1"
    for i in "${!MODEL_KEYS[@]}"; do
        if [ "${MODEL_KEYS[$i]}" = "$key" ]; then
            echo "${MODEL_REPOS[$i]}"
            return 0
        fi
    done
    return 1
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

# --- Select which ASR model to run -------------------------------------------
# One of the keys in ASR_MODEL_REPOS (see src/models.py), e.g. v3, persian-v4,
# persian-bf16, qwen3-asr-1.7b. Override non-interactively by exporting
# ASR_MODEL before running this script, e.g.:
#   ASR_MODEL=qwen3-asr-1.7b ./deploy.sh
# Otherwise, when run interactively, you'll be prompted to pick from the
# models already downloaded into ./hf-cache (see download.sh).
if [ -z "${ASR_MODEL+x}" ] && [ -n "${WHISPER_MODEL+x}" ]; then
    ASR_MODEL="$WHISPER_MODEL"
fi
if [ -z "${ASR_MODEL+x}" ] && [ -t 0 ] && [ "${#MODEL_KEYS[@]}" -gt 0 ]; then
    echo "Available ASR models:"
    for i in "${!MODEL_KEYS[@]}"; do
        key="${MODEL_KEYS[$i]}"
        status="not downloaded"
        is_model_downloaded "${MODEL_REPOS[$i]}" && status="downloaded"
        printf "  %d) %-16s %-45s family=%-10s [%s]\n" "$((i+1))" "$key" "${MODEL_REPOS[$i]}" "${MODEL_FAMILY[$key]:-whisper}" "$status"
    done
    read -rp "Select a model to use [1-${#MODEL_KEYS[@]}] (default: persian-v4): " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#MODEL_KEYS[@]}" ]; then
        ASR_MODEL="${MODEL_KEYS[$((choice-1))]}"
    fi
fi
export ASR_MODEL="${ASR_MODEL:-persian-v4}"
# Keep WHISPER_MODEL in sync for backward compatibility with anything still
# reading that env var (src/models.py also accepts either).
export WHISPER_MODEL="$ASR_MODEL"
echo "Active ASR model: $ASR_MODEL (family: ${MODEL_FAMILY[$ASR_MODEL]:-whisper})"

# Extract container names from docker-compose.yml (one per service)
mapfile -t CONTAINER_NAMES < <(grep "container_name:" docker-compose.yml | awk '{print $2}')
echo "Container names from docker-compose.yml: ${CONTAINER_NAMES[*]}"

# Check if any of the containers exist and stop/remove them via compose
NEEDS_DOWN=false
for name in "${CONTAINER_NAMES[@]}"; do
    if [ "$(docker ps -aq -f name="^${name}\$")" ]; then
        NEEDS_DOWN=true
        break
    fi
done

if [ "$NEEDS_DOWN" = true ]; then
    echo "Existing containers found, stopping and removing them..."
    docker compose down
else
    echo "No existing containers found, will build new ones."
fi

# Build the image (without starting) so we can download the selected model
# into the shared hf-cache volume before the service actually loads it.
echo "Building image using docker compose..."
docker compose build

SELECTED_REPO="$(repo_for_key "$ASR_MODEL" || true)"
if [ -n "$SELECTED_REPO" ] && is_model_downloaded "$SELECTED_REPO"; then
    echo "ASR model '$ASR_MODEL' ($SELECTED_REPO) already downloaded, skipping download."
elif [ -n "$SELECTED_REPO" ]; then
    echo "Downloading ASR model '$ASR_MODEL' ($SELECTED_REPO) into the shared cache..."
    docker compose run --rm ai-tts python3 -c "from src.models import download_asr_model; download_asr_model('$ASR_MODEL')"
else
    echo "Warning: '$ASR_MODEL' is not a known key in ASR_MODEL_REPOS (src/models.py); skipping pre-download."
fi


# Start containers using docker compose
echo "Starting containers using docker compose..."
docker compose up -d

# Verify all containers are running
ALL_RUNNING=true
for name in "${CONTAINER_NAMES[@]}"; do
    if [ "$(docker ps -q -f name="^${name}\$")" ]; then
        echo "Container $name is now running."
    else
        echo "Failed to start container $name."
        ALL_RUNNING=false
    fi
done

if [ "$ALL_RUNNING" != true ]; then
    exit 1
fi

echo "Deployment completed successfully."
