
PROJECT_DIR="$(dirname "$(readlink -f "$0")")"

# Navigate to project directory
cd "$PROJECT_DIR"
echo "Deploying from directory: $PROJECT_DIR"

# Extract container name from docker-compose.yml
CONTAINER_NAME=$(grep "container_name:" docker-compose.yml | awk '{print $2}')
echo "Container name from docker-compose.yml: $CONTAINER_NAME"

# Check if container exists and is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME is running, stopping and removing it..."
    docker compose down
elif [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME exists but is not running, removing it..."
    docker compose down
else
    echo "Container $CONTAINER_NAME does not exist, will build a new one."
fi

# Build and start container using docker compose
echo "Building and starting container using docker compose..."
docker compose up -d --build

# Verify container is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME is now running."
else
    echo "Failed to start container $CONTAINER_NAME."
    exit 1
fi

echo "Deployment completed successfully."
