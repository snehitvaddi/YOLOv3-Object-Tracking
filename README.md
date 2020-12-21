## Object Tracking using YOLOv3, Deepsort and Tensorflow
This repository implements YOLOv3 and Deep SORT in order to perfrom real-time object tracking. Yolov3 is an algorithm that uses deep convolutional neural networks to perform object detection. We can feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order for a real-time object tracker to be created.

![Demo of Object Tracker](data/video/traffic-result-gif.gif)
### Author:  <i><small> I referenced this repo from Youtube video by [The AI Guy](https://www.youtube.com/watch?v=Cf1INvUsvkM&lc=z225j1ixysjxwhlvnacdp431jphj0oobdzwbosngo0dw03c010c.1585682883809851) </small></i>


|🧠 Original Repo|💡 Colab Notebook|
|--------|---------|
|[Github](https://github.com/theAIGuysCode/yolov3_deepsort)| [Colab](https://colab.research.google.com/drive/1PrEt-t-uLXgA8k8eeSn3SrSsnZlXS3Br)|

### Note : If you are running this on Google Colab, output .avi file is beeing corrupted. I am still figuring it out why? and will update once I did it. For now I suggest to try it on local machine.

## Getting started

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate tracker-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU, if you haven't set it up already)
```bash
# Ubuntu 18.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```
### Downloading official pretrained weights
<strong>For Linux</strong>: 
You can download official yolov3 weights pretrained on COCO dataset.
```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights
```
<strong>For Windows</strong>:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and yolov3-tiny [here](https://pjreddie.com/media/files/yolov3-tiny.weights) then save them to the weights folder.

### Using Custom trained weights
<strong> Learn How To Train Custom YOLOV3 Weights Here: https://www.youtube.com/watch?v=zJDUhGL26iU </strong>

Add your custom weights file to weights folder and your custom .names file into data/labels folder.
  
### Saving your yolov3 weights as a TensorFlow model.
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .tf model files!

```
# yolov3
python load_weights.py

# yolov3-tiny
python load_weights.py --weights ./weights/yolov3-tiny.weights --output ./weights/yolov3-tiny.tf --tiny

# yolov3-custom (add --tiny flag if your custom weights were trained for tiny model)
python load_weights.py --weights ./weights/<YOUR CUSTOM WEIGHTS FILE> --output ./weights/yolov3-custom.tf --num_classes <# CLASSES>
```

After executing one of the above lines, you should see proper .tf files in your weights folder. You are now ready to run object tracker.

## Running the Object Tracker
Now you can run the object tracker for whichever model you have created, pretrained, tiny, or custom.
```
# yolov3 on video
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi

#yolov3 on webcam 
python object_tracker.py --video 0 --output ./data/video/results.avi

#yolov3-tiny 
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi --weights ./weights/yolov3-tiny.tf --tiny

#yolov3-custom (add --tiny flag if your custom weights were trained for tiny model)
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi --weights ./weights/yolov3-custom.tf --num_classes <# CLASSES> --classes ./data/labels/<YOUR CUSTOM .names FILE>
```
The output flag saves your object tracker results as an avi file for you to watch back. It is not necessary to have the flag if you don't want to save the resulting video.

There is a test video uploaded in the data/video folder called test.mp4. If you have followed all the steps properly then you should see the output as below by running first command.ie;
```
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi
```
#### Video Example
![Demo of Object Tracker](data/helpers/demo.gif)

## Acknowledgments
* [Yolov3 TensorFlow Amazing Implementation](https://github.com/zzh8829/yolov3-tf2)
* [Deep SORT Repository](https://github.com/nwojke/deep_sort)
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
