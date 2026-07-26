FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/home/user0/uda_env/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash user0

COPY uda_env.tar.gz /tmp/uda_env.tar.gz

RUN mkdir -p /home/user0/uda_env \
    && tar -xzf /tmp/uda_env.tar.gz -C /home/user0/uda_env \
    && rm /tmp/uda_env.tar.gz \
    && /home/user0/uda_env/bin/conda-unpack \
    && chown -R user0:user0 /home/user0

USER user0
WORKDIR /home/user0/app
