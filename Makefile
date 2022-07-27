up:
	docker-compose up -d --remove-orphans

down:
	docker-compose stop 

restart:
	docker-compose stop ; docker-compose up -d --remove-orphans;

backend: 
	docker exec -it wholetensor-backend bash

frontend: 
	docker exec -it wholetensor-frontend sh

subtensor: 
	docker exec -it wholetensor-subtensor sh
