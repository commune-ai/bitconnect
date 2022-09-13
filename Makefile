up:
	docker-compose up -d --remove-orphans

down:
	docker-compose stop 

restart:
	docker-compose stop ; docker-compose up -d --remove-orphans;

app:
	docker exec -it wholetensor-backend bash -c "streamlit run commune/bittensor/module.py"

bash:
	docker exec -it wholetensor-${arg} bash

kill_all:
	docker kill $(docke ps -q)

prune:
	docker rm $(docker ps --filter status=exited -q)