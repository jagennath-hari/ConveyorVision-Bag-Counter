from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

class testData:
    def __init__(self):
        self.model = YOLO('/home/hari/cement/runs/detect/train/weights/best.pt')

    def predict(self, img):
        result = self.model.predict(img)
        result = result[0]
        img = Image.fromarray(result.plot()[:, :, : : -1]).convert('RGB')
        open_cv_image = np.array(img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        cv2.imshow("RESULT", open_cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    test = testData()
    test.predict("/home/hari/cement/datasets/images/00118.jpg")

if __name__ == "__main__":
    main()