import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List

class videoTest:
    def __init__(self, video_path, model_path):
        self.cap = cv2.VideoCapture(str(video_path))
        self.model = YOLO(str(model_path))

    def detections2boxes(self, detections: Detections) -> np.ndarray:
        return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
            ))
    # converts List[STrack] into format that can be consumed by match_detections_with_tracks function
    def tracks2boxes(self, tracks: List[STrack]) -> np.ndarray:
        return np.array([
            track.tlbr
            for track
            in tracks
        ], dtype=float)
    
    # matches our bounding boxes with predictions
    def match_detections_with_tracks(self, detections: Detections, tracks: List[STrack]) -> Detections:
        if not np.any(detections.xyxy) or len(tracks) == 0:
            return np.empty((0,))

        tracks_boxes = self.tracks2boxes(tracks=tracks)
        iou = self.box_iou_batch(tracks_boxes, detections.xyxy)
        track2detection = np.argmax(iou, axis=1)

        tracker_ids = [None] * len(detections)

        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                tracker_ids[detection_index] = tracks[tracker_index].track_id

        return tracker_ids

    def processVideo(self):
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")
            # Read until video is completed
        while(self.cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret == True:
            
                # Display the resulting frame
                cv2.imshow('Result', self.deployModel(frame))
            
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Break the loop
            else: 
                break
            
            # When everything done, release the video capture object
        self.cap.release()
            
            # Closes all the frames
        cv2.destroyAllWindows()

    def deployModel(self, img):
        result = self.model.predict(img)
        result = result[0]
        img = Image.fromarray(result.plot()[:, :, : : -1]).convert('RGB')
        open_cv_image = np.array(img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        return open_cv_image



def main():
    test = videoTest("/home/hari/cement/datasets/172.20.6.226_Truck Loading PP - 1_main_20230722095635.mp4", "/home/hari/cement/runs/detect/train/weights/best.pt")
    test.processVideo()

if __name__ == "__main__":
    main()