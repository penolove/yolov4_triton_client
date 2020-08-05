FROM nvcr.io/nvidia/tritonserver:20.07-py3-clientsdk

COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt