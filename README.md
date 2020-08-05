# yolov4_triton_client

we will provide a yolov4 triton client example in this repo

the yolov4 tensorrt engine were generated from: https://github.com/wang-xinyu/tensorrtx
which is a awesome repo with lots of famous network tensorrt implmentation

pre-requirement:

- docker
- triton server serving with yolov4 [TODO](provide the meidum link)

## run a client

```bash
docker build -t yolov4_triton_client .

docker run -ti --net host -v $(pwd):/tmp/script yolov4_triton_client /bin/bash;

python /tmp/script/simple_grpc_infer_client.py
```

## TODO

- complete the args of simple_grpc_infer_client