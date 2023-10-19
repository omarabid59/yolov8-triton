import numpy as np
import json
import triton_python_backend_utils as pb_utils
import cv2


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        num_detections_config = pb_utils.get_output_config_by_name(
            model_config, "num_detections")
        detection_boxes_config = pb_utils.get_output_config_by_name(
            model_config, "detection_boxes")

        detection_scores_config = pb_utils.get_output_config_by_name(
            model_config, "detection_scores")

        detection_classes_config = pb_utils.get_output_config_by_name(
            model_config, "detection_classes")

        # Convert Triton types to numpy types
        self.num_detections_dtype = pb_utils.triton_string_to_numpy(
            num_detections_config['data_type'])

        # Convert Triton types to numpy types
        self.detection_boxes_dtype = pb_utils.triton_string_to_numpy(
            detection_boxes_config['data_type'])

        # Convert Triton types to numpy types
        self.detection_scores_dtype = pb_utils.triton_string_to_numpy(
            detection_scores_config['data_type'])

        # Convert Triton types to numpy types
        self.detection_classes_dtype = pb_utils.triton_string_to_numpy(
            detection_classes_config['data_type'])

        self.score_threshold = 0.25
        self.nms_threshold = 0.45

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        num_detections_dtype = self.num_detections_dtype
        detection_boxes_dtype = self.detection_boxes_dtype
        detection_scores_dtype = self.detection_scores_dtype
        detection_classes_dtype = self.detection_classes_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

            # Get the output arrays from the results
            outputs = in_0.as_numpy()

            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)
                 ) = cv2.minMaxLoc(classes_scores)
                if maxScore >= self.score_threshold:
                    box = [outputs[0][i][0] -
                           (0.5 *
                            outputs[0][i][2]), outputs[0][i][1] -
                           (0.5 *
                            outputs[0][i][3]), outputs[0][i][2], outputs[0][i][3]]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            result_boxes = cv2.dnn.NMSBoxes(boxes, scores,
                                            self.score_threshold,
                                            self.nms_threshold,
                                            0.5)

            num_detections = 0
            output_boxes = []
            output_scores = []
            output_classids = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    'class_id': class_ids[index],
                    'confidence': scores[index],
                    'box': box}
                output_boxes.append(box)
                output_scores.append(scores[index])
                output_classids.append(class_ids[index])

                num_detections += 1

            num_detections = np.array(num_detections)
            num_detections = pb_utils.Tensor(
                "num_detections", num_detections.astype(num_detections_dtype))

            detection_boxes = np.array(output_boxes)
            detection_boxes = pb_utils.Tensor(
                "detection_boxes", detection_boxes.astype(detection_boxes_dtype))

            detection_scores = np.array(output_scores)
            detection_scores = pb_utils.Tensor(
                "detection_scores", detection_scores.astype(detection_scores_dtype))
            detection_classes = np.array(output_classids)
            detection_classes = pb_utils.Tensor(
                "detection_classes",
                detection_classes.astype(detection_classes_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    num_detections,
                    detection_boxes,
                    detection_scores,
                    detection_classes])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
