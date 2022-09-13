#/bin/bash



eval nohup ./scripts/gradio/api.sh &> nohup/gradio_api.out &
eval nohup ./scripts/gradio/client.sh &> nohup/gradio_client.out &
