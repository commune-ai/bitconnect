down:
	./start.sh --backend --frontend --down
stop:
	make down
up:
	./start.sh --backend --frontend
start:
	make up

up_latest:
	./start.sh --backend

start_latest:
	./start.sh --backend

bash_backend: 
	docker exec -it bittensor-backend bash

frontend:
	docker exec -it frontend bash

backend: 
	docker exec -it luca-bittensor-backend bash

restart:
	make down && make up;


prune_volumes:	
	docker system prune --all --volumes

bash:
	docker exec -it bittensor-${arg} bash

app:
	make streamlit
kill_all:
	docker kill $(docker ps -q)

logs:
	docker logs ${arg} --tail=100 --follow

streamlit:
	docker exec -it luca-bittensor-backend bash -c "streamlit run commune/${arg}/module.py "
enter_backend:
	docker exec -it bittensor-backend bash
pull:
	git submodule update --init --recursive
	
kill_all:
	docker kill $(docker ps -q) 

jupyter:
	docker exec -it bittensor-backend bash -c "jupyter lab --allow-root --ip=0.0.0.0 --port=8888"

python:
	docker exec -it bittensor-backend bash -c "python commune/${arg}/module.py"

exec:
	docker exec -it bittensor-backend bash -c "${arg}"

api:
	docker exec -it luca-bittensor-backend bash -c "python commune/gradio/api/module.py --api"


gradio_api:
	docker exec -it bittensor-backend bash -c "python commune/gradio/api/module.py --api"

gradio_client:
	make app arg=gradio/client

