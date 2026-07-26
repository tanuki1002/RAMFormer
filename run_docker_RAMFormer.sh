#!/usr/bin/env bash
set -e

ENV_NAME="uda"
TARBALL="uda_env.tar.gz"
IMAGE_NAME="ramformer"
CONTAINER_NAME="ramformer"

# 1. Pack the conda environment (skip if already packed)
if [ ! -f "$TARBALL" ]; then
    echo ">> Packing conda environment '$ENV_NAME' into $TARBALL ..."
    conda-pack -n "$ENV_NAME" -o "$TARBALL"
fi

# 2. Build the docker image
docker build -t "$IMAGE_NAME" -f dockerfile .

# 3. Run the container (mounts current project dir, enables GPU)
docker run -it \
    --name="$CONTAINER_NAME" \
    --gpus=all \
    --shm-size=32g \
    --volume="$(pwd):/home/user0/app" \
    "$IMAGE_NAME" /bin/bash
