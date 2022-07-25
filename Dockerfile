FROM nvcr.io/nvidia/pytorch:21.06-py3
ARG host_uid
WORKDIR /workspace
COPY requirements.txt ./requirements.txt

RUN conda install faiss-gpu
RUN pip install -r requirements.txt

RUN useradd -u $host_uid -ms /bin/bash -G root tczinczoll
USER tczinczoll