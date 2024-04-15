#!/usr/bin/env bash
USAGE=$(cat << EOF
Usage:
  -m, --message  Commit message
  -h, --help: Show this help message
EOF
)

while [ $# -gt 0 ]; do
  case "$1" in
    -m|--message)
      shift
      MESSAGE=$1
      ;;
    --help|-h)
      echo "$USAGE"
      exit 0
      ;;
    *)
      echo "Error: Invalid argument\n"
      exit 1
      ;;
  esac
  shift
done

if [ -z "$MESSAGE" ]; then
  echo "Error: Missing commit message\n"
  exit 1
fi

rm *.zip
if [ ! -d "packages" ]; then
    pip download -r requirements.txt -d packages
fi
zip -r -j  packages.zip packages/*
zip -r -j  models.zip models/*
zip -r -j  configs.zip configs/*
zip -r -j  torch-cache.zip $HOME/.cache/torch/*
kaggle datasets version -p . -m "$MESSAGE"
