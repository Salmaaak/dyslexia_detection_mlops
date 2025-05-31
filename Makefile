.PHONY: test train deploy

test:
	docker-compose run test

train:
	docker-compose run train

deploy:
	docker-compose down
	docker-compose up -d --build