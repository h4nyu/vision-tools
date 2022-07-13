#!/bin/sh
zip -r dist.zip . -x checkpoints/**\* lightning_logs/**\* .mypy_cache/**\* .pytest_cache/**\* __pycache__/**\*
