# Overview
This repository provides an ensemble model to combine a YoloV8 model exported from the [Ultralytics](https://github.com/ultralytics/ultralytics) repository with NMS post-processing. The NMS post-processing code contained in [models/postprocess/1/model.py](models/postprocess/1/model.py) is adapted from the [Ultralytics ONNX Example](https://github.com/ultralytics/ultralytics/blob/4b866c97180842b546fe117610869d3c8d69d8ae/examples/YOLOv8-OpenCV-ONNX-Python/main.py).


For more information about Triton's Ensemble Models, see their documentation on [Architecture.md](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md) and some of their [preprocessing examples](https://github.com/triton-inference-server/python_backend/tree/main/examples/preprocessing).

# Directory Structure
```
models/
    yolov8_onnx/
        1/
            model.onnx
        config.pbtxt
        
    postprocess/
        1/
            model.py
        config.pbtxt
        
    yolov8_ensemble/
        1/
            <Empty Directory>
        config.pbtxt
README.md
main.py
```


# Quick Start
1. Install [Ultralytics](https://github.com/ultralytics/ultralytics) and TritonClient
```
pip install ultralytics==8.0.51 tritonclient[all]==2.31.0
```

2. Export a model to ONNX format:
```
yolo export model=yolov8n.pt format=onnx dynamic=True opset=16
```

3. Rename the model file to `model.onnx` and place it under the `/models/yolov8_onnx/1` directory (see directory structure above).

4. (Optional): Update the Score and NMS threshold in [models/postprocess/1/model.py](models/postprocess/1/model.py#L59)

5. (Optional): Update the [models/yolov8_ensemble/config.pbtxt](models/yolov8_ensemble/config.pbtxt) file if your input resolution has changed.

6. Build the Docker Container for Triton Inference:
```
DOCKER_NAME="yolov8-triton"
docker build -t $DOCKER_NAME .
```

6. Run Triton Inference Server:
```
DOCKER_NAME="yolov8-triton"
docker run --gpus all \
    -it --rm \
    --net=host \
    -v ./models:/models \
    $DOCKER_NAME
```

7. Run the script with `python main.py`. The overlay image will be written to `output.jpg`.



