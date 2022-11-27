FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --yes
RUN apt-get install --yes tmux python3-dev python3-pip

# build nvtop
RUN git clone https://github.com/Syllo/nvtop.git /tmp/nvtop && \
    cd /tmp/nvtop && \
    mkdir build && cd build \
    cmake .. && make && make install

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -U pip wheel
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
ENTRYPOINT [ "/bin/bash" ]
