# https://hub.docker.com/_/python/

DOCKER_IMAGE=python:3.8.12
CONTAINER_NAME=dementia_ai
DOCKER_ROOT=/opt/aible
PWD_APPLICATION=$DOCKER_ROOT/applications/$CONTAINER_NAME
PWD_SHARED=$DOCKER_ROOT/shared
PWD_DATA=$PWD_APPLICATION/data
PWD_LOG=$PWD_APPLICATION/log
PWD_UPLOAD=$DOCKER_ROOT/upload

OPTS="
        -itd
        --restart=unless-stopped
        --log-opt max-size=10m
        -w /usr/src/myapp
"

VOLUMES="
        -v $PWD_SHARED/localtime:/etc/localtime
        -v $PWD_SHARED/timezone:/etc/timezone
        -v $PWD_APPLICATION/startup.sh:/startup.sh
        -v $PWD_DATA/work:/usr/src/myapp
        -v $PWD_UPLOAD:/upload
        -v $PWD_LOG:/log
        -v $PWD_DATA/python3.8:/usr/local/lib/python3.8
"

PORTS="
        -p 9090:9090
"

ENVS="
"

docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
docker run $OPTS $VOLUMES $PORTS $ENVS --name $CONTAINER_NAME $DOCKER_IMAGE sh /startup.sh
