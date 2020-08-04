FROM nvcr.io/nvidia/tritonserver:20.07-v1-py3-clientsdk

COPY requirments.txt /requirments.txt
RUN pip3 install -r /requirments.txt