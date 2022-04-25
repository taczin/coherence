FROM nvcr.io/nvidia/pytorch:22.03-py3
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

