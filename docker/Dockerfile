FROM  scrin/dev-spconv:latest

RUN pip3 install easydict \
					hdbscan \
					numba \
					numpy \
					pyyaml \
					scikit-learn \
					scipy \
					tensorboard \
					#torch=1.5 \
					#torchvision=0.6.0 \ 
					#torch-cluster=1.5.7 \
					#torch-scatter=1.3.2  \
					tqdm
RUN pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html


# #install required libs and packages
# RUN apt-get update  \
# 	&& apt-get install -y --no-install-recommends wget build-essential git curl ca-certificates libjpeg-dev libpng-dev python3-pip \
# 	&& rm -rf /var/lib/apt/lists/*

# ENV PYTHON_VERSION=3.6
# ENV FORCE_CUDA="1"
# ENV CUDA cu110
# # CUDA path
# ENV PATH=/usr/local/cuda-11.0/bin:$PATH
# ENV CUDA_PATH=/usr/local/cuda-11.0
# ENV CUDA_HOME=/usr/local/cuda-11.0
# ENV LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# # CuDNN installation 
# # You need to download cudnn.tar file from  https://developer.nvidia.com/rdp/cudnn-download
# COPY deps/cudnn-11.0-linux-x64-v8.0.4.30.tgz cudnn-11.0-linux-x64-v8.0.4.30.tgz    
# RUN tar -xzf cudnn-11.0-linux-x64-v8.0.4.30.tgz
# RUN cp cuda/include/cudnn*.h /usr/local/cuda/include
# RUN cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
# RUN chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
# RUN rm -rf cudnn-11.0-linux-x64-v8.0.4.30.tgz
# RUN rm -rf cuda

# # ROS Noetic installation
# ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y lsb-release && apt-get clean all
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
# RUN apt-get update  \
# 	&& apt-get install -y ros-noetic-desktop
# RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
# RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
# RUN /bin/bash -c "source ~/.bashrc"

# # Update pip & install pytorch
# RUN python3 -m pip install pip --upgrade
# RUN python3 -m pip install wheel
# RUN pip install pyyaml numpy ipython
# ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# RUN pip install torch torchvision 

# # CMake installation
# RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.16.4/cmake-3.16.4-Linux-x86_64.tar.gz
# RUN tar -xzf cmake-3.16.4-Linux-x86_64.tar.gz --exclude=bin/cmake-gui --exclude=doc/cmake --exclude=share/cmake-3.12/Help  \
# 	&& cp -fR cmake-3.16.4-Linux-x86_64/* /usr  \
# 	&& rm -rf cmake-3.16.4-Linux-x86_64  \
# 	&& rm cmake-3.16.4-Linux-x86_64.tar.gz

# # SpConv installation
# RUN git clone https://github.com/traveller59/spconv.git --recursive
# RUN cd spconv  \
# #	&& git checkout 7342772  \
# 	&& python3 setup.py bdist_wheel  \
# 	&& cd ./dist  \
# 	&& pip install *  \
# 	&& cd ../..  \
# 	&& rm -r spconv

# # # Apex installation
# WORKDIR /tmp/unique_for_apex
# RUN git clone https://github.com/NVIDIA/apex.git  \
#  	&& cd apex \
#  	&& git checkout 5633f6
# WORKDIR /tmp/unique_for_apex/apex
# RUN python3 setup.py install
# RUN rm -rf /tmp/unique_for_apex

# # installing ROS and centerpoint packages
# RUN apt install -y ros-noetic-pcl-msgs ros-noetic-pcl-ros ros-noetic-image-geometry ros-noetic-tf2 ros-noetic-tf-conversions
# RUN	pip install rospkg scikit-image scipy numba matplotlib fire protobuf opencv-python opencv-contrib-python pybind11 vtk joblib \
# 	easydict open3d-python terminaltables pytest-runner addict pycocotools imagecorruptions objgraph cachetools descartes \
# 	motmetrics pandas Pillow pyquaternion scikit-learn Shapely tqdm requests empy rosdep

# RUN git clone https://github.com/eric-wieser/ros_numpy.git \
# 	&& cd ros_numpy  \
# 	&& python3 setup.py install \
# 	&& cd .. \ 
# 	&& rm -r ros_numpy

# # ARG USER=docker_centerpoint
# # ARG UID=1000
# # ARG GID=1000
# # # default password
# # ARG PW=user 

# # RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

# # WORKDIR /home/${USER}
# # RUN cd /home/${USER} && mkdir -p catkin_ws/src && chown -R ${UID}:${GID} ./

# # USER ${UID}:${GID}
