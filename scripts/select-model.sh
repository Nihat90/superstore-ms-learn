az ml model create \
  --name superstore-loss-rf \
  --path azureml://jobs/DEIN_BESTER_CHILD_JOB/outputs/model_output/paths/model.joblib \
  --type custom_model \
  --resource-group "DEINE_RESOURCE_GROUP" \
  --workspace-name "DEIN_WORKSPACE"
