import os
import io
import av
import numpy as np
import cv2
import ultralytics


class FaceDetector():
    def __init__(self, 
                 model_path='best.onnx',
                 device='cpu',
                 crops_path=None,
                 meta_path=None,
                 persist=True,
                 conf=0.23,
                 iou=0.8,
                 tracker='bytetrack.yaml') -> None:
        """
        Initialize the FaceDetector object.

        Args:
            model_path (str): Path to the model file.
            device (str): Specifies the device for inference (e.g., cpu, cuda:0 or 0).
            crops_path (str): Best bboxes of faces crops will be saved there
            meta_path (str): Meta data for each frame of video will be saved there
            persist (bool): Argument tells the tracker that the current image or frame is the next in a sequence and to expect tracks from the previous image in the current image.
            conf (float): Sets the minimum confidence threshold for detections.
            iou (float): Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS).
            tracker (str): botsort.yaml/bytetrack.yaml (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers)

        Returns:
            None
        """
        ultralytics.checks()
        self.model = ultralytics.YOLO(model_path, task='detect')
        self.device = device
        self.crops_path = crops_path
        self.meta_path = meta_path
        self.tracker_params = {
            "device": device,
            "persist": persist,
            "conf": conf,
            "iou": 0.8,
            "tracker": tracker
        }
        self.frame_number = 0
        if crops_path is not None:
            # self.best_crops contains id: (confidence score, cropped_bbox)
            self.best_crops = {}
    
    def _save_crops(self, img, tracks):
        
        
    def _save_meta(self, tracks):
        
        
    def update(self, frame):
        """
        Updates tracker and returns Results class object

        Parameters:
              img (numpy.ndarray): Input image containing faces.

        Returns:
              results (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.
        """
        results = self.model.track(frame, **self.tracker_params)
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        if self.crops_path: 
            self._save_crops(frame, track_ids)
        if self.meta_path:
            self._save_meta(tracks_ids)
        self.frame_number += 1
        return results
    
    def get_results(self, img):
        """
        Inferences the detector and returns Results class object

        Parameters:
              img (numpy.ndarray): Input image containing faces.

        Returns:
              results (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.
        """
        return self.model()
    
    def __call__(self, img):
        """
        Calls the draw_box method to draw bounding boxes around detected faces in the input image.

        Parameters:
              img (numpy.ndarray): Input image containing faces.

        Returns:
              output_img (matplotlib.pyplot.figure): Image with bounding boxes plotted around detected faces.
        """
        return self.draw_box(img)
