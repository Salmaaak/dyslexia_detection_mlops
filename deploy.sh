#!/bin/bash
docker pull $DOCKERHUB_USERNAME/dyslexia-detector:latest
docker-compose down
docker-compose up -d