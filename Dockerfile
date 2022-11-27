FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --yes
RUN apt-get install --yes tmux python3-dev python3-pip git cmake make build-essential

# build nvtop
RUN pip3 install --no-cache-dir -U pip wheel
RUN pip3 install --no-cache-dir jupyter notebook
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
ENTRYPOINT [ "/bin/bash" ]
CMD [ "jupyter notebook --ip=0.0.0.0 --port=7777 --allow-root --NotebookApp.token='mytoken' --NotebookApp.password=''" ]
