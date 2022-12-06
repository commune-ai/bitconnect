down:
	./scripts/start.sh --down
stop:
	make down
up:
	./scripts/start.sh --light
start:
	./scripts/start.sh --${arg} 
logs:
	./scripts/start.sh --${arg}

build:
	./scripts/start.sh --build --${arg}

bash_backend: 
	make bash arg=backend

restart:
	make down && make up;

prune_volumes:	
	docker system prune --all --volumes

bash:
	docker exec -it ${arg} bash

app:
	make streamlit

kill_all:
	docker kill $(docker ps -q)

logs:
	docker logs ${arg} --tail=100 --follow

streamlit:
	docker exec -it backend bash -c "streamlit run ${arg}.py "
	
enter_backend:
	docker exec -it backend bash

pull:
	git submodule update --init --recursive
	
kill_all:
	docker kill $(docker ps -q) 

python:
	docker exec -it backend bash -c "python ${arg}.py"

exec:
	docker exec -it backend bash -c "${arg}"