#!/bin/bash
set -e

CMD="jupyter notebook --ip=0.0.0.0 --port=7777 --allow-root --NotebookApp.token='mytoken' --NotebookApp.password=''"
docker build -t sinfusion_pytorch_unofficial .
docker run --rm -it sinfusion_pytorch_unofficial -v $(pwd):/workspace -c "$CMD"
