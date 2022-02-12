SHELL=/bin/bash
IMAGE=drltetris:latest

mrproper:
	docker-compose build --no-cache

build-image:
	docker-compose build

build: build-image

flush-redis:
	docker exec -it drl-tetris_redis_1 redis-cli FLUSHALL
