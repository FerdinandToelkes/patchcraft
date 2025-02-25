FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# ffmpeg libsm6 libxext6 -> needed for script usage?? 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y --no-install-recommends \
    python3.10-dev \
    python3-pip \
    libffi-dev \
    build-essential \
    rsync 
     
# create a directory for the package source code, -p flag creates the directory if it doesn't exist
RUN mkdir -p /patchcraft

# set the working directory 
WORKDIR /patchcraft/src

# copy the requirements.text file needed to install the package to the set working directory
COPY requirements.txt .

# upgrade pip
RUN python3 -m pip install --upgrade pip

# install the requirements
RUN pip3 install -r requirements.txt

# keep container running
CMD tail -f /dev/null




