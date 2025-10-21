FROM nvcr.io/nvidia/pytorch:23.12-py3

LABEL MAINTAINER=yuu

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /code/mbr

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-pip git default-jre cpanminus ffmpeg curl jq && \
    pip3 install six && \
    cpanm --force XML::Parser && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install git-lfs for handling large files
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /code/mbr

COPY requirements.txt /code/mbr/requirements.txt

# Install all regular packages at once with uv
RUN uv venv
RUN uv pip install -r requirements.txt --system

COPY mbr /code/mbr/mbr

COPY experiments /code/mbr/experiments

RUN mkdir -p /code/mbr/results

ENTRYPOINT ["/bin/bash"]