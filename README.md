some object detection code

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

