import argparse
import numpy as np
import sys

import tritongrpcclient
from eyewitness.config import BoundedBoxObject

from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs


BBOX_CONF_THRESH = 0.5


def preprocess(pil_image_obj, input_image_shape=(608, 608)):
    """
    since the tensorRT engine with a fixed input shape, and we don't want to resize the
    original image directly, thus we perform a way like padding and resize the original image
    to align the long side to the tensorrt input
    Parameters
    ----------
    pil_image_obj: PIL.image.object

    input_image_shape: tuple[int]
        H, W shpae of the tensorrt model
    Returns
    -------
    image: np.array
        np.array with shape: NCHW, value between 0~1
    scale_ratio: float
        scale down ratio
    """
    height_scale_weight = input_image_shape[0] / pil_image_obj.height
    width_scale_weight = input_image_shape[1] / pil_image_obj.width

    scale_ratio = min(width_scale_weight, height_scale_weight)
    image_resized_shape = tuple(
        int(i * scale_ratio) for i in [pil_image_obj.width, pil_image_obj.height]
    )

    output_img = np.zeros((1, 3, *input_image_shape))
    processed_image = resize_and_stack_image_objs(
        image_resized_shape, [pil_image_obj]
    )  # NHWC
    processed_image = np.transpose(processed_image, [0, 3, 1, 2])  # NCHW

    # insert the processed image into the empty image
    output_img[
        :, :, : image_resized_shape[1], : image_resized_shape[0]
    ] = processed_image

    # Convert the image to row-major order, also known as "C order"
    output_img = np.array(output_img, dtype=np.float32, order="C")
    output_img /= 255.0  # normalize
    return output_img, scale_ratio


def nms(boxes, box_confidences, iou_threshold=0.5):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
    confidence scores and return an array with the indexes of the bounding boxes we want to
    keep (and display later).

    Parameters
    ----------
    boxes: np.ndarray
        NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
    box_confidences: np.ndarray
        a Numpy array containing the correspohnding confidences with shape N
    iou_threshold: float
        a threshold between (0, 1)

    Returns
    -------
    selected_index: List[int]
        the selected index to keep
    """
    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(
            x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]]
        )
        yy2 = np.minimum(
            y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]]
        )

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = areas[i] + areas[ordered[1:]] - intersection

        # Compute the Intersection over Union (IoU) score:
        iou = intersection / union

        # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
        # candidates to a minimum. In this step, we keep only those elements whose overlap
        # with the current bounding box is lower than the threshold:
        indexes = np.where(iou <= iou_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep


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
        "--img",
        type=str,
        required=False,
        default="/tmp/script/samples/bus.jpg",
    )
    parser.add_argument(
        "--drawn_img",
        type=str,
        required=False,
        default="/tmp/script/samples/drawn_image.jpg",
    )

    FLAGS = parser.parse_args()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "yolov4"

    # Infer
    inputs = []
    outputs = []
    # the built engine with input NCHW
    inputs.append(tritongrpcclient.InferInput("data", [1, 3, 608, 608], "FP32"))

    # Initialize the data
    image_obj = Image("image_id", raw_image_path=FLAGS.img)
    ori_w, ori_h = image_obj.pil_image_obj.size
    image_frame, scale_ratio = preprocess(
        image_obj.pil_image_obj, input_image_shape=(608, 608)
    )
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
    n_bbox = int(output0_data[0, 0, 0, 0])
    bbox_matrix = output0_data[0, 1: (n_bbox * 7 + 1), 0, 0].reshape(-1, 7)

    detected_objects = []
    if n_bbox:
        labels = set(bbox_matrix[:, 5])
        for label in labels:
            idices = np.where(
                (bbox_matrix[:, 5] == label) & (bbox_matrix[:, 6] >= BBOX_CONF_THRESH)
            )
            sub_bbox_matrix = bbox_matrix[idices]
            box_confidences = bbox_matrix[idices, 6]
            keep_idices = nms(sub_bbox_matrix[:, :4], sub_bbox_matrix[:, 6])
            sub_bbox_matrix = sub_bbox_matrix[keep_idices]

            for idx in range(sub_bbox_matrix.shape[0]):
                x, y, w, h, _, label, score = sub_bbox_matrix[idx, :]
                x1 = (x - w / 2) / scale_ratio
                x2 = min((x + w / 2) / scale_ratio, ori_w)
                y1 = (y - h / 2) / scale_ratio
                y2 = min((y + h / 2) / scale_ratio, ori_h)
                if x1 == x2:
                    continue
                if y1 == y2:
                    continue
                detected_objects.append(
                    BoundedBoxObject(x1, y1, x2, y2, label, score, "")
                )

    ImageHandler.draw_bbox(image_obj.pil_image_obj, detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, FLAGS.drawn_img)
