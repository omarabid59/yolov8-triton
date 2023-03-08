import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import sys
import argparse


def get_triton_client(url: str = 'localhost:8001'):
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=False,
            keepalive_options=keepalive_options)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    return triton_client


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'({class_id}: {confidence:.2f})'
    color = (255, 0, )
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def read_image(image_path: str) -> np.ndarray:
    original_image: np.ndarray = cv2.imread(image_path)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    input_image = cv2.resize(image, (640, 640))
    input_image = (input_image / 255.0).astype(np.float32)

    # Channel first
    input_image = input_image.transpose(2, 0, 1)

    # Expand dimensions
    input_image = np.expand_dims(input_image, axis=0)
    return original_image, input_image, scale


def run_inference(model_name: str, input_image: np.ndarray,
                  triton_client: grpcclient.InferenceServerClient):
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('images', input_image.shape, "FP32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(input_image)

    outputs.append(grpcclient.InferRequestedOutput('num_detections'))
    outputs.append(grpcclient.InferRequestedOutput('detection_boxes'))
    outputs.append(grpcclient.InferRequestedOutput('detection_scores'))
    outputs.append(grpcclient.InferRequestedOutput('detection_classes'))

    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    num_detections = results.as_numpy('num_detections')
    detection_boxes = results.as_numpy('detection_boxes')
    detection_scores = results.as_numpy('detection_scores')
    detection_classes = results.as_numpy('detection_classes')
    return num_detections, detection_boxes, detection_scores, detection_classes


def main(image_path, model_name, url):
    triton_client = get_triton_client(url)
    original_image, input_image, scale = read_image(image_path)
    num_detections, detection_boxes, detection_scores, detection_classes = run_inference(
        model_name, input_image, triton_client)

    for index in range(num_detections):
        box = detection_boxes[index]

        draw_bounding_box(original_image,
                          detection_classes[index],
                          detection_scores[index],
                          round(box[0] * scale),
                          round(box[1] * scale),
                          round((box[0] + box[2]) * scale),
                          round((box[1] + box[3]) * scale))

    cv2.imwrite('output.jpg', original_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./assets/bus.jpg')
    parser.add_argument('--model_name', type=str, default='yolov8_ensemble')
    parser.add_argument('--url', type=str, default='localhost:8001')
    args = parser.parse_args()
    main(args.image_path, args.model_name, args.url)
