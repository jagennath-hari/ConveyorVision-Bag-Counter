import cv2
import os

class frameExtractor:
    def __init__(self, video_path, write_path):
        self.cap = cv2.VideoCapture(str(video_path))
        self.write_path = str(write_path)

    def showVideo(self, frame_interval=50):
        frame_count = 29500
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                frame_count += 1
                if frame_count % frame_interval == 0:
                    image_name = "{:05d}.jpg".format(int(frame_count / frame_interval))
                    print(os.path.join(self.write_path, image_name))
                    cv2.imwrite(os.path.join(self.write_path, image_name), frame)
                    frame = cv2.resize(frame, (0, 0), None, fx = 0.5, fy = 0.4)
                    cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    extract = frameExtractor("datasets/bag_upright.mp4", "/home/hari/cement/datasets/images")
    extract.showVideo()

if __name__ == "__main__":
    main()