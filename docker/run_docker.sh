HERE=$(pwd) # Absolute path of current directory

user=`whoami`
uid=`id -u`
gid=`id -g`

#echo "$user $uid $gid"

DOCKER_REPO="dimdano/"

BRAND="adapt-cpu"
VERSION="0.1"

IMAGE_NAME=${DOCKER_REPO}$BRAND:${VERSION}

docker run \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $HERE:/workspace/adapt \
    -w /workspace/adapt \
    -it \
    --rm \
    --network=host \
    $IMAGE_NAME \
    bash

