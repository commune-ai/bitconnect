up:
	docker-compose up -d --remove-orphans

down:
	docker-compose stop 

restart:
	docker-compose stop ; docker-compose up -d --remove-orphans;

backend: 
	docker exec -it wholetensor-backend bash

app:
	docker exec -it wholetensor-backend bash -c "streamlit run commune/bittensor/module.py"
frontend: 
	docker exec -it wholetensor-frontend sh

subtensor: 
	docker exec -it wholetensor-subtensor sh

bash:
	docker exec -it wholetensor-${arg} bash