#!/bin/bash

cd /home/viovna/Documents/stealth/commune-ai/app/backend/src
# run the graphql api
docker exec -it -d backend uvicorn graph_ql.main:app --reload --port 8000 --host 0.0.0.0
## run the apps
#nohup streamlit run run_app_user.py --server.port 8501 &> endpoint_app_user.out&
#nohup streamlit run run_app_experiment.py --server.port 8502 &> endpoint_app_experiment.out&
#nohup streamlit run run_app_preprocess.py --server.port 8503 &> endpoint_app_preprocess.out&
