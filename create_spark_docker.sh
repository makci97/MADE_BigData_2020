docker run -d \
    --hostname jupyterhub-toree \
    --log-opt max-size=50m \
    -p 8000:8000 \
    -p 4040:4040 \
    -e DOCKER_USER=$(id -un) \
    -e DOCKER_USER_ID=$(id -u) \
    -e DOCKER_PASSWORD=$(id -un) \
    -e DOCKER_GROUP_ID=$(id -g) \
    -e DOCKER_ADMIN_USER=$(id -un) \
    -v /Users/ruaihm4/code:/workdir \
    -v $(dirname $HOME):/home_host \
    dclong/jupyterhub-toree /scripts/sys/init.sh
