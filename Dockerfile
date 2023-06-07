# * Copyright (c) 2009-2020. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

# FROM nvidia/cuda:11.4-base

# FROM nvidia/cuda:11.4.0-base-ubuntu18.04
FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
CMD nvidia-smi

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
# RUN --gpus all nvidia/cuda:11.4.0-base-ubuntu18.04 nvidia-smi

FROM cytomine/software-python3-base:v2.2.0
# Install Stardist and tensorflow
RUN pip3 install tensorflow-gpu==2.8.0
RUN pip3 install stardist==0.8.0
RUN pip3 install protobuf==3.20.*

#INSTALL
RUN pip3 install numpy
RUN pip3 install shapely
RUN pip3 install tifffile
RUN pip3 install torch
RUN pip3 install torchvision


RUN mkdir -p /models 
ADD 3333nuclei_densenet_best_model_100ep.pth /models/3333nuclei_densenet_best_model_100ep.pth
ADD 22k_nuclei_densenet21_best_model_100ep.pth /models/22k_nuclei_densenet21_best_model_100ep.pth
RUN chmod 444 /models/3333nuclei_densenet_best_model_100ep.pth
RUN chmod 444 /models/22k_nuclei_densenet21_best_model_100ep.pth

RUN cd /models && \
    mkdir -p 2D_versatile_HE
ADD 2D_versatile_HE/config.json /models/2D_versatile_HE/config.json
ADD 2D_versatile_HE/thresholds.json /models/2D_versatile_HE/thresholds.json
ADD 2D_versatile_HE/weights_best.h5 /models/2D_versatile_HE/weights_best.h5
RUN chmod 444 /models/2D_versatile_HE/config.json
RUN chmod 444 /models/2D_versatile_HE/thresholds.json
RUN chmod 444 /models/2D_versatile_HE/weights_best.h5

RUN cd /models && \
    mkdir -p 2D_versatile_fluo
ADD 2D_versatile_fluo/config.json /models/2D_versatile_fluo/config.json
ADD 2D_versatile_fluo/thresholds.json /models/2D_versatile_fluo/thresholds.json
ADD 2D_versatile_fluo/weights_best.h5 /models/2D_versatile_fluo/weights_best.h5
RUN chmod 444 /models/2D_versatile_HE/config.json
RUN chmod 444 /models/2D_versatile_HE/thresholds.json
RUN chmod 444 /models/2D_versatile_HE/weights_best.h5

#ADD FILES
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD stardistpndensenet.py /app/stardistpndensenet.py

ENTRYPOINT ["python3", "/app/stardistpndensenet.py"]
