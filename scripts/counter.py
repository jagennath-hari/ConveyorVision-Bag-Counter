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
from onemetric.cv.utils.iou import box_iou_batch

class BYTETrackerArgs:
    track_thresh: float = 0.50
    track_buffer: int = 1000
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

class videoTest:
    def __init__(self, video_path, model_path):
        self.cap = cv2.VideoCapture(str(video_path))
        self.model = YOLO(str(model_path))
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        LINE_START = Point(561, 517)
        LINE_END = Point(906, 510)
        self.line_counter = LineCounter(start=LINE_START, end=LINE_END)

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
        iou = box_iou_batch(tracks_boxes, detections.xyxy)
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
                #print(self.deployModel(frame))
                cv2.imshow("FRAME", self.deployModel(frame))
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
            # Break the loop
            else: 
                break
            
            # When everything done, release the video capture object
        self.cap.release()
            
            # Closes all the frames
        cv2.destroyAllWindows()

    def deployModel(self, img):
        LINE_START = Point(50, 1500)
        LINE_END = Point(3840-50, 1500)
        CLASS_NAMES_DICT = self.model.names
        CLASS_ID = [0]
        results = self.model.predict(img)
        detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int))

        tracks = self.byte_tracker.update(
            output_results=self.detections2boxes(detections=detections),
            img_info=img.shape,
            img_size=img.shape
        )


        tracker_id = self.match_detections_with_tracks(detections=detections, tracks=tracks)

        detections.tracker_id = np.array(tracker_id)

        # format custom labels
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        self.line_counter.update(detections=detections)


        # annotate and display frame
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
        frame = box_annotator.annotate(frame=img, detections=detections, labels=labels)

        line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
        line_annotator.annotate(frame=frame, line_counter=self.line_counter)

        return frame



def main():
    test = videoTest("/home/hari/cement/datasets/172.20.6.226_Truck Loading PP - 1_main_20230722112747.mp4", "/home/hari/cement/runs/detect/train/weights/best.pt")
    test.processVideo()

if __name__ == "__main__":
    main()