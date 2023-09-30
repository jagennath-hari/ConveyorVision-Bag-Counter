# ConveyorVision-Bag-Counter

## 📄 Abstract
<div align="justify">
ConveyCount is an innovative real-time system designed to automate the counting and tracking of cement bags on conveyor belts. Utilizing cutting-edge deep learning techniques like YOLOv8 for object detection and Byte tracker for precise tracking, ConveyCount accurately monitors cement bags as they traverse the conveyor belt. 
</div>

## 🏁 Dependencies
1) NVIDIA Driver ([Official Download Link](https://www.nvidia.com/download/index.aspx))
2) CUDA Toolkit ([Official Link](https://developer.nvidia.com/cuda-downloads))
3) Miniconda ([Official Link](https://docs.conda.io/en/main/miniconda.html))
4) PyTorch ([Official Link](https://pytorch.org/))
5) Ultralytics YOLOv8 ([Official Link](https://github.com/ultralytics/ultralytics))
6) ByteTracker ([Official Link](https://github.com/ifzhang/ByteTrack))
7) Supervision ([Official Link](https://github.com/roboflow/supervision))
8) Onemetric ([Official Link](https://github.com/SkalskiP/onemetric))

## ⚙️ Install
1) Create conda env
2) Install dependencies into env
3) Annotate your datasets of cement bag. A good online data annotation tool is [Roboflow](https://roboflow.com) or [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/). A `data.yaml` file must get created along with `train`, `valid` and  `test` folders containing the images and labels.
4) Follow [Official Link](https://docs.ultralytics.com) to train network and generate `yolo8.pt` file with your network architecture of choice, along with your dataset.


## 🪪 License 
See the [LICENSE](LICENSE) file for details.
