FROM  scrin/dev-spconv:latest

RUN pip3 install easydict \
					hdbscan \
					numba \
					numpy \
					pyyaml \
					scikit-learn \
					scipy \
					tensorboard \
					tqdm
RUN pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html


ARG USER=docker_centerpoint
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user 

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

WORKDIR /home/${USER}
RUN cd /home/${USER} && mkdir -p catkin_ws/src && chown -R ${UID}:${GID} ./

USER ${UID}:${GID}
