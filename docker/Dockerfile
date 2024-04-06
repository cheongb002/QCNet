FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
LABEL org.opencontainers.image.authors="john.lee@robotics.utias.utoronto.ca"

ENV TZ Canada/Eastern
ENV DEBIAN_FRONTEND noninteractive

SHELL ["/bin/bash", "-c"]

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN echo "Username is $USERNAME"
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

#? System Packages
COPY apt_packages.txt apt_packages.txt
RUN apt-get update && \
    xargs -a apt_packages.txt apt-get install -y --no-install-recommends && \
    rm apt_packages.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip;

#? Install basic python packages
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt && \
    rm requirements.txt
RUN pip3 install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.2.2%2Bcu121.html

#? Add shortcut for jupyter notebook
RUN echo "alias jn='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'" >> /home/$USERNAME/.bashrc

USER $USERNAME
WORKDIR /home/$USERNAME