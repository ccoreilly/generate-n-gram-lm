FROM ubuntu:focal-20210416

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget \
    git \
    g++ \
    cmake \
    python3 python3-pip \ 
    # for kenlm
    libboost-thread-dev libboost-test-dev libboost-system-dev libboost-program-options-dev \
    zlib1g-dev libbz2-dev liblzma-dev && \
    # ==================================================================
    # clean up everything
    # ------------------------------------------------------------------
    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && git checkout 0c4dd4e && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/kenlm \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j$(nproc)

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && rm requirements.txt

WORKDIR /opt/kenlm