az ml model create \
  --name superstore-loss-rf \
  --path azureml://jobs/DEIN_BESTER_CHILD_JOB/outputs/model_output/paths/model.joblib \
  --type custom_model \
  --resource-group rg-ai300-l453a46dc08054eb79a \
  --workspace-name mlw-ai300-l453a46dc08054eb79a
