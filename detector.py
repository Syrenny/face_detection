import os
import json
import io
import av
import numpy as np
import cv2
import ultralytics
from PIL import Image


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
            crops_path (str): Best bboxes of faces crops will be saved in this directory. If not exist, it will be created.
            meta_path (str): Path to .json file. If not exist, it will be created.
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
            if not os.path.exists(self.crops_path):
                os.makedirs(self.crops_path)
            # self.best_crops contains id: (max confidence score)
            self.best_crops = {}
        if meta_path is not None:
            self.meta = {}
            # Проверяем, существует ли директория по указанному пути
            if not os.path.exists(os.path.dirname(meta_path)):
                # Если директория отсутствует, создаем ее
                os.makedirs(os.path.dirname(meta_path))

    def _crop_image(self, image, bbox):
        """
        Crops bbox from image

        Parameters:
              image (numpy.ndarray): Input image containing face.
              bbox (numpy.ndarray): Bbox in xyxy.
        """
        bbox = np.int32(bbox)
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def _save_crop(self, image, bbox, track_id):
        """
        Saves the crop of face in <self.crop_path>/<track_id>/<track_id>_best_crop.jpg

        Parameters:
              image (numpy.ndarray): Input image containing face.
              bbox (numpy.ndarray): Bbox in xyxy.
              track_id (int): Tracker index of current bbox.
        """
        new_image = Image.fromarray(self._crop_image(image, bbox))
        image_path = os.path.join(self.crops_path, str(track_id), f"{track_id}_best_crop.jpg")
        if os.path.exists(image_path):
            existing_image = Image.open(os.path.join(image_path))
            existing_image.paste(new_image)
            existing_image.save(image_path)
        else:
            os.makedirs(os.path.join(self.crops_path, str(track_id)))
            new_image.save(image_path)
            
    def _update_crops(self, results):
        """
        Updates self.best_crops, which contains the best confidence score for each tracker index.
        Saves crop when new bbox with higher confidence score was found.

        Parameters:
              results (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.
        """
        tracks = results[0].boxes.id.int().cpu().tolist()
        bboxes = results[0].boxes.cpu().xyxy
        confidence = results[0].boxes.cpu().conf
        for i, (track_id, xyxy, conf) in enumerate(zip(tracks, bboxes, confidence)):
            if track_id in self.best_crops: 
                if conf > self.best_crops[track_id]:
                    self._save_crop(results[0].orig_img, xyxy, track_id)
                    self.best_crops[track_id] = conf
            else: 
                self.best_crops[track_id] = conf 
                self._save_crop(results[0].orig_img, xyxy, track_id)

    def _update_meta(self, results):
        """
        Saves xyxy and confidence of current bboxes in .json file self.meta_path

        Parameters:
              results (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.
        """
        frame_meta = {}
        tracks = results[0].boxes.id.int().cpu().tolist()
        bboxes = results[0].boxes.cpu().xyxy
        confidence = results[0].boxes.cpu().conf
        
        for i, (track_id, xyxy, conf) in enumerate(zip(tracks, bboxes, confidence)):
            track_meta = {"xyxy": xyxy.tolist(), "confidence": float(conf)}
            frame_meta[track_id] = track_meta
        self.meta[self.frame_number] = frame_meta
         
        with open(self.meta_path, 'w') as file:
            json.dump(self.meta, file)

    def update(self, frame):
        """
        Updates tracker and returns Results class object

        Parameters:
              img (numpy.ndarray): Input image containing faces.

        Returns:
              results (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.
        """
        results = self.model.track(frame, **self.tracker_params)
        if results[0].boxes.id is not None:
            if self.crops_path is not None: 
                self._update_crops(results)
            if self.meta_path is not None:
                self._update_meta(results)
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
        return self.get_results(img)
