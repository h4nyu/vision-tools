[![circleci](https://circleci.com/gh/h4nyu/vision-tools.svg?style=svg)](https://app.circleci.com/pipelines/github/h4nyu/vision-tools?filter=all)
[![codecov](https://codecov.io/gh/h4nyu/vision-tools/branch/master/graph/badge.svg?token=TLYBISJIE4)](https://codecov.io/gh/h4nyu/vision-tools)

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

| Packages                                   | Description  |
| :-                                         | :-           |
| **[vision_tools](./vision_tools)**         | lib          |
| **[kuzushiji_bench](./kuzushiji_bench)**   | kuzushiji    |
| **[cots_bench](./cots_bench)**        | Detect crown-of-thorns starfish in underwater image data    |
