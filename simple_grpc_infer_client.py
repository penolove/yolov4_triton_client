import argparse
import numpy as np
import sys

import tritongrpcclient
from eyewitness.config import BoundedBoxObject

from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-c',
                        '--use_custom_model',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use custom model')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "yolov4" 

    # Infer
    inputs = []
    outputs = []
    inputs.append(tritongrpcclient.InferInput('data', [1, 3, 608, 608], "FP32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    # input_data = np.ones(shape=(1, 3, 608, 608), dtype=np.float32)

    # Initialize the data
    image_obj = Image('image_id', raw_image_path='/tmp/script/samples/bus.jpg')
    processed_image = resize_and_stack_image_objs(
        (608, 608), [image_obj.pil_image_obj])
    input_data = np.transpose(processed_image, [0, 3, 1, 2]).astype(np.float32) / 255.0

    inputs[0].set_data_from_numpy(input_data)

    outputs.append(tritongrpcclient.InferRequestedOutput('prob'))

    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs,
                                  headers={'test': '1'})

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    print(statistics)

    # Get the output arrays from the results
    output0_data = results.as_numpy('prob')
    n_bbox = int(output0_data[0, 0, 0, 0])
    bbox_matrix = output0_data[0, 1:(n_bbox * 7+1), 0, 0].reshape(-1, 7)

    detected_objects = []
    if n_bbox:
        # TODO resacle
        for idx in range(n_bbox):
            x, y, w, h, _, label, score = bbox_matrix[idx, :]

            x1 = x - w / 2
            x2 = x + w / 2
            y1 = y - h / 2
            y2 = y + h / 2
            if x1 == x2:
                continue

            if y1 == y2:
                continue
            detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, label, score, ''))

    import ipdb; ipdb.set_trace()
    resized_img = image_obj.pil_image_obj.resize((608, 608))
    ImageHandler.draw_bbox(resized_img, detected_objects)
    ImageHandler.save(resized_img, "drawn_image.jpg")