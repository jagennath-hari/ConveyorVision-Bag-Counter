# ConveyorVision-Bag-Counter

## 📄 Abstract
<div align="justify">
ConveyCount is an innovative real-time system designed to automate the counting and tracking of cement bags on conveyor belts. Utilizing cutting-edge deep learning techniques like YOLOv8 for object detection and Byte tracker for precise tracking, ConveyCount accurately monitors cement bags as they traverse the conveyor belt. Its seamless integration, reliable counting at the referee line, and robust performance in complex environments make it a valuable tool for optimizing industrial processes and enhancing productivity.
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
1) Create conda env.
2) Install dependencies into env.
3) Annotate your datasets of cement bags. A good online data annotation tool is [Roboflow](https://roboflow.com) or [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/). A `data.yaml` file must get created along with `train`, `valid` and  `test` folders containing the images and labels.
4) Follow [Official Link](https://docs.ultralytics.com) to train network and generate `yolo8.pt` file with your network architecture of choice, along with your dataset.

## 🤖 To Use
1) Update the `video file` and `.pt` file paths in `counter.py` in the `main()` function.
2) Run `python counter.py` inside your conda env.

## 📊 Result
<div align="center">
    <img src="assets/conveyorvision_output.gif" alt="SLAM" width="700"/>
    <p>SLAM</p>
</div>

## 📑 Report
A brief [REPORT](assets/ConveyorVision.pdf) can be read to better understand the algorithm.

## 🪪 License 
See the [LICENSE](LICENSE) file for details.
