DOCKERFILE="AdaPT.Dockerfile"
VERSION="0.1"

# Build Image Tag
IMAGE_TAG="dimdano/adapt-cpu:${VERSION}"

echo "Building AdaPT image..."

docker build -f $DOCKERFILE --tag=$IMAGE_TAG .


