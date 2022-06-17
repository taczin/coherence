FROM nvcr.io/nvidia/pytorch:21.06-py3

COPY requirements.txt ./requirements.txt

RUN conda install faiss-gpu
RUN pip install -r requirements.txt
