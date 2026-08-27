
PROJECT_DIR="$(dirname "$(readlink -f "$0")")"

# Navigate to project directory
cd "$PROJECT_DIR"
echo "Deploying from directory: $PROJECT_DIR"

HF_CACHE_DIR="$PROJECT_DIR/hf-cache/transformers"

# --- Whisper model discovery -------------------------------------------------
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

mapfile -t MODEL_ENTRIES < <(parse_whisper_models)
MODEL_KEYS=()
MODEL_REPOS=()
for entry in "${MODEL_ENTRIES[@]}"; do
    MODEL_KEYS+=("${entry%% *}")
    MODEL_REPOS+=("${entry#* }")
done

# --- Select which Whisper model to run ---------------------------------------
# One of the keys in WHISPER_MODELS (see src/models.py), e.g. v3, persian-v4,
# persian-bf16. Override non-interactively by exporting WHISPER_MODEL before
# running this script, e.g.:
#   WHISPER_MODEL=v3 ./deploy.sh
# Otherwise, when run interactively, you'll be prompted to pick from the
# models already downloaded into ./hf-cache (see download.sh).
if [ -z "${WHISPER_MODEL+x}" ] && [ -t 0 ] && [ "${#MODEL_KEYS[@]}" -gt 0 ]; then
    echo "Available Whisper models:"
    for i in "${!MODEL_KEYS[@]}"; do
        status="not downloaded"
        is_model_downloaded "${MODEL_REPOS[$i]}" && status="downloaded"
        printf "  %d) %-14s %-45s [%s]\n" "$((i+1))" "${MODEL_KEYS[$i]}" "${MODEL_REPOS[$i]}" "$status"
    done
    read -rp "Select a model to use [1-${#MODEL_KEYS[@]}] (default: persian-v4): " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#MODEL_KEYS[@]}" ]; then
        WHISPER_MODEL="${MODEL_KEYS[$((choice-1))]}"
    fi
fi
export WHISPER_MODEL="${WHISPER_MODEL:-persian-v4}"
echo "Active Whisper model: $WHISPER_MODEL"

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

SELECTED_REPO="$(repo_for_key "$WHISPER_MODEL" || true)"
if [ -n "$SELECTED_REPO" ] && is_model_downloaded "$SELECTED_REPO"; then
    echo "Whisper model '$WHISPER_MODEL' ($SELECTED_REPO) already downloaded, skipping download."
elif [ -n "$SELECTED_REPO" ]; then
    echo "Downloading Whisper model '$WHISPER_MODEL' ($SELECTED_REPO) into the shared cache..."
    docker compose run --rm ai-tts python3 -c "from src.models import download_whisper_model; download_whisper_model('$SELECTED_REPO')"
else
    echo "Warning: '$WHISPER_MODEL' is not a known key in WHISPER_MODELS (src/models.py); skipping pre-download."
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
