down:
	./start.sh --backend --down
stop:
	make down
up:
	./start.sh --backend --update
start:
	make up

bash_backend: 
	docker exec -it wholetensor-backend bash

restart:
	./start.sh --all --restart;


prune_volumes:	
	docker system prune --all --volumes

bash:
	docker exec -it wholetensor-${arg} bash

app:
	make streamlit
kill_all:
	docker kill $(docker ps -q)

logs:
	docker logs ${arg} --tail=100 --follow

streamlit:
	docker exec -it wholetensor-backend bash -c "streamlit run commune/${arg}/module.py "
enter_backend:
	docker exec -it wholetensor-backend bash
pull:
	git submodule update --init --recursive
	
kill_all:
	docker kill $(docker ps -q) 

jupyter:
	docker exec -it wholetensor-backend bash -c "jupyter lab --allow-root --ip=0.0.0.0 --port=8888"

python:
	docker exec -it wholetensor-backend bash -c "python algocean/${arg}/module.py"

exec:
	docker exec -it wholetensor-backend bash -c "${arg}"

