some object detection code
[![circleci](https://circleci.com/gh/h4nyu/vision-tools.svg?style=svg)](https://app.circleci.com/pipelines/github/h4nyu/vision-tools?filter=all)


## Setup

#### For container developers

1. Manually setup `.env`. 

```
COMPOSE_FILE=docker-compose.yaml:docker-compose.gpu.yaml
TENSORBOARD_PORT=8008
```

2. Build local enviroment image

```
docker-compose build app
```

