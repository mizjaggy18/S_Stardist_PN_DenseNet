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
FROM nvidia/cuda:11.4.0-base-ubuntu18.04
RUN apt update
# CMD nvidia-smi


FROM cytomine/software-python3-base:v2.2.0
# Install Stardist and tensorflow
RUN pip install tensorflow-gpu==2.8.0
RUN pip install stardist==0.8.0
RUN pip install protobuf==3.20.*

#INSTALL
RUN pip install numpy
RUN pip install shapely
RUN pip install tifffile
RUN pip install torch
RUN pip install torchvision

RUN mkdir -p /models 
ADD 3333nuclei_densenet_best_model_100ep.pth /models/3333nuclei_densenet_best_model_100ep.pth
RUN chmod 444 /models/3333nuclei_densenet_best_model_100ep.pth

RUN cd /models && \
    mkdir -p 2D_versatile_HE
ADD config.json /models/2D_versatile_HE/config.json
ADD thresholds.json /models/2D_versatile_HE/thresholds.json
ADD weights_best.h5 /models/2D_versatile_HE/weights_best.h5
RUN chmod 444 /models/2D_versatile_HE/config.json
RUN chmod 444 /models/2D_versatile_HE/thresholds.json
RUN chmod 444 /models/2D_versatile_HE/weights_best.h5


#ADD FILES
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD stardistpndensenet.py /app/stardistpndensenet.py

ENTRYPOINT ["python3", "/app/stardistpndensenet.py"]
