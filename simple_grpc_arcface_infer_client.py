import argparse
import sys

import numpy as np
import tritongrpcclient
from sklearn import preprocessing
from eyewitness.image_utils import Image, resize_and_stack_image_objs


def preprocess(pil_image_obj, input_image_shape=(112, 112)):
    processed_image = resize_and_stack_image_objs(
        input_image_shape, [pil_image_obj])

    processed_image = (processed_image) / 256
    processed_image = np.transpose(processed_image, [0, 3, 1, 2])  # NCHW
    return processed_image.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )

    parser.add_argument(
        "--img1",
        type=str,
        required=False,
        default="/tmp/script/samples/joey0.ppm",
    )

    parser.add_argument(
        "--img2",
        type=str,
        required=False,
        default="/tmp/script/samples/joey1.ppm",
    )

    FLAGS = parser.parse_args()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "arcface"

    # Infer
    inputs = []
    outputs = []
    
    # the built engine with input NCHW
    inputs.append(tritongrpcclient.InferInput("data", [1, 3, 112, 112], "FP32"))

    # Initialize the data
    image_obj = Image("image_id", raw_image_path=FLAGS.img1)
    image_frame = preprocess(image_obj.pil_image_obj)
    inputs[0].set_data_from_numpy(image_frame)
    outputs.append(tritongrpcclient.InferRequestedOutput("prob"))

    # Test with outputs
    results = triton_client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs, headers={"test": "1"}
    )

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    print(statistics)

    # Get the output arrays from the results
    output0_data = results.as_numpy("prob")

    # image2
    inputs = []
    outputs = []
    # the built engine with input NCHW
    inputs.append(tritongrpcclient.InferInput("data", [1, 3, 112, 112], "FP32"))

    # Initialize the data
    image_obj = Image("image_id", raw_image_path=FLAGS.img2)
    image_frame = preprocess(image_obj.pil_image_obj)
    inputs[0].set_data_from_numpy(image_frame)
    outputs.append(tritongrpcclient.InferRequestedOutput("prob"))

    # Test with outputs
    results = triton_client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs, headers={"test": "1"}
    )

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    print(statistics)

    # Get the output arrays from the results
    output1_data = results.as_numpy("prob").reshape(-1)

    image_embedding = preprocessing.normalize(
        np.stack([output0_data.reshape(-1), output1_data.reshape(-1)]), axis=1)

    print(image_embedding.dot(image_embedding.T))