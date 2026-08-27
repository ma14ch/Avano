
PROJECT_DIR="$(dirname "$(readlink -f "$0")")"

# Navigate to project directory
cd "$PROJECT_DIR"
echo "Deploying from directory: $PROJECT_DIR"

# Which Whisper model to use at runtime. One of: v3, persian-v4
# (see WHISPER_MODELS in src/models.py). Override by exporting
# WHISPER_MODEL before running this script, e.g.:
#   WHISPER_MODEL=v3 ./deploy.sh
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

# Build the image (without starting) so we can pre-download both models
# into the shared hf-cache volume before the service actually loads one.
echo "Building image using docker compose..."
docker compose build

echo "Pre-downloading all Whisper models into the shared cache..."
docker compose run --rm ai-tts python3 -c "from src.models import download_all_whisper_models; download_all_whisper_models()"

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
