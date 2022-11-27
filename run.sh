#!/bin/bash
set -e

CMD="jupyter notebook --ip=0.0.0.0 --port=7777 --allow-root --NotebookApp.token='mytoken' --NotebookApp.password=''"
docker build -t rgabdullin/sinfusion_pytorch_unofficial:dev .
docker push rgabdullin/sinfusion_pytorch_unofficial:dev
# docker run --rm -it -v $(pwd):/workspace -p 7777:7777 sinfusion_pytorch_unofficial -c "$CMD"
