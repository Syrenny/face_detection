import io
import numpy as np
import cv2
import onnxruntime
import ultralytics
from ultralytics import YOLO


# Local packages

ultralytics.checks()

      
    
class Yolov8Detector():

    def __init__(self, model_path='best.onnx', size=(640, 640), classes={0: 'face'}) -> None:
        """
        Initialize the MyDetector object.

        Args:
            model_path (str): Path to the model file.
            size (tuple): Size of the input image.

        Returns:
            None
        """
        self._size = size
        self.classes = classes
        self.model = YOLO(model_path, task='detect')
    

    def video_run(self, cap):
        """
        Process video frames and draw bboxes.

        Args:
            cap: Video capture object.

        Returns:
            io.BytesIO: In-memory file containing the processed video.
        """
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        
        # Video stream handling 
        output_memory_file = io.BytesIO()
        output_f = av.open(output_memory_file, 'w', format='mp4')  # Open "in memory file" as MP4 video output
        stream = output_f.add_stream('h264', str(fps))  # Add H.264 video stream to the MP4 container, with framerate = fps.
        stream.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stream.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Video capturing
        while True:
            ret, frame = cap.read()
            if not ret:
                break    
            #find bounding boxes and track them via YOLO model method    
            results = self.model.track(frame,device='cpu',  persist=True, conf=0.23, iou = 0.8, tracker='bytetrack.yaml')
#             results = self.model.predict(frame, conf=0.23 , iou=0.8)
            
            #draw bboxes
            annotated_frame = results[0].plot()
#             print(annotated_frame.shape)
            # Convert image from NumPy Array to frame.
            annotated_frame = av.VideoFrame.from_ndarray(annotated_frame, format='bgr24') 
            packet = stream.encode(annotated_frame)  # Encode video frame
            output_f.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).
            
        # Flush the encoder
        packet = stream.encode(None)
        output_f.mux(packet)
        output_f.close()
        return output_memory_file
    
    def draw_box(self, img):
        """
        This method is responsible for drawing bounding boxes around detected faces in an image.
        
        Parameters:
              img (numpy.ndarray): Input image containing faces.
              
        Returns:
              output_img (matplotlib.pyplot.figure): Image with bounding boxes drawn around detected faces.
        """
        output = self.model.predict(img, device='cpu', conf= 0.23, iou = 0.8)
        output_img = output[0].plot()
        return output_img
    
    
    def __call__(self, img):
        """
        Calls the draw_box method to draw bounding boxes around detected faces in the input image.

        Parameters:
              img (numpy.ndarray): Input image containing faces.

        Returns:
              output_img (matplotlib.pyplot.figure): Image with bounding boxes plotted around detected faces.
        """
        return self.draw_box(img)
