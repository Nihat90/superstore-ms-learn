#!/bin/bash

set -e

az ml job create \
  --file configs/train-job.yml \
  --resource-group "DEINE_RESOURCE_GROUP" \
  --workspace-name "DEIN_WORKSPACE"
