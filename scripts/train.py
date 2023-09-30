from ultralytics import YOLO

class modelTraining:
    def __init__(self):
        self.model = YOLO('yolov8n.yaml')
        self.model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
        self.model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    def train(self):
        self.model.train(data='/home/hari/cement/data.yaml', epochs=1000, imgsz=640)

def main():
    trainer = modelTraining()
    trainer.train()

if __name__ == "__main__":
    main()