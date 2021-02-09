# YOLOv4-object: an efficient model and method for obejct discovery

[![Darknet](https://img.shields.io/badge/Powered%20by-Darknet-green)](https://github.com/AlexeyAB/darknet) [![Darknet](https://img.shields.io/badge/Train%20on-Colab-yellow)](https://colab.research.google.com/drive/11jaSKfF74bPPVO0-9o-ZgLE5yVWUeI3d?usp=sharing)

This repo is based on [Darknet](https://github.com/AlexeyAB/darknet).
The model YOLOv4-object aims at discovering all object instances without explicit class definition.

The comparison between object detection and object discovery in our project is shown below:
<p align="left">
  <img src="https://github.com/forever208/YOLOv4-object/blob/master/data/demo.png" width='100%' height='100%'/>
</p>

our results is shown in the table below:
| Models | Meta-model | Class amount | Class names | mAP@50 | BFlops | FPS (Tesla V100) | 
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| YOLOv4 | -      | 80 | person, car, chair... | 62.08% | 60.1 | 101 |
| YOLOv4-object | YOLOv4 | 1 | object | 69.08% | 59.6 | 104 |
| YOLOv4-human | YOLOv4 | 1 | human | 79.37% | 59.6 | 104 |
| YOLOv4-human-object | YOLOv4 | 2 | human, object | 71.04% | 59.6 | 104 |
| YOLOv4-object(slim) | YOLOv4 | 1 | object | 69.00% | 51.4 | 115 |
| YOLOv4-object(finetune) | YOLOv4 | 1 | object | 65.77% | 59.6 | 104 |
| YOLOv4-tiny | - | 80 | person, car, chair... | 40.20% | 6.9 | 505 |
| YOLOv4-tiny-object | YOLOv4-tiny | 1 | object | 45.16% | 6.8 | 523 |
| YOLOv4-tiny-object (slim) | YOLOv4-tiny | 1 | object | 45.53% | 6.7 | 558 |

- We are gonna introduce the way of running, training, testing your own model in Linux.
- For macOS and windows users, refer to [Darknet](https://github.com/AlexeyAB/darknet) is suggested.
- we also recommend you to run this repo in Google Colab for saving the time of environment setting. A step-by-step introdution is included in each Jupyter Notebook ([YOLOv4-object](https://colab.research.google.com/drive/11jaSKfF74bPPVO0-9o-ZgLE5yVWUeI3d?usp=sharing), [YOLOv4-object (finefune)](https://colab.research.google.com/drive/17QSl3Eh3d1-MQH4xTpzMhUCTSLhSj7mN?usp=sharing), [YOLOv4-object (slim)](https://colab.research.google.com/drive/1zUucq9y5NeTvI5E7hCF4p2tTAh-TvZwO?usp=sharing), [YOLOv4-human-object](https://colab.research.google.com/drive/1N_hlL21sLejqjiJ-b9KXavcZ_guknmt3?usp=sharing)). If you run into problems in Colab, feel free to check this [tutorial](https://www.youtube.com/watch?v=mmj3nxGT2YQ) out   


### 1. Install Darknet (Linux) 
clone Darknet
```sh
$ git clone https://github.com/AlexeyAB/darknet
$ cd darknet
```

change the Makefile for using GPU
```sh
$ sed -i 's/OPENCV=0/OPENCV=1/' Makefile
$ sed -i 's/GPU=0/GPU=1/' Makefile
$ sed -i 's/CUDNN=0/CUDNN=1/' Makefile
$ sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```

build darknet
```sh
$ make
```

download yolov4.weights and test YOLOv4 on your images
```sh
$ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
$ ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights <img_path> -thresh 0.3
```

### 2. Run YOLOv4-object
clone this repo
```sh
$ cd ..
$ git clone https://github.com/forever208/YOLOv4-object.git
$ cd darknet
```
copy cfg files into darknet folder
```sh
$ cp /YOLOv4-object/cfg/yolov4-obj.cfg ./cfg
$ cp /YOLOv4-object/cfg/obj.names ./data
$ cp /YOLOv4-object/cfg/obj.data ./data
```
Download our weights file (yolov4-obj_6577.weights) 
```sh
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-Z3nHBPEWpAEJ8-PkNKAyt1K9oB3FtNX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-Z3nHBPEWpAEJ8-PkNKAyt1K9oB3FtNX" -O train2017_txts_universal.zip && rm -rf /tmp/cookies.txt
```
Run the model YOLOv4-obejct on images, video or test its FPS on your computer.

**test on images**
```sh
$ ./darknet detector test <your obj.data> <your config> <your weights file> <your image> -thresh 0.3
# for example
$ ./darknet detector test data/obj.data cfg/yolov4-obj.cfg yolov4-obj_6577.weights /YOLOv4-object/data/000000029596.jpg -thresh 0.3
```

**test on videos**
```sh
$ ./darknet detector demo <your obj.data> <your config> <your weights file> -dont_show <your video> -i 0 -thresh 0.3 -out_filename <output path>
# for example
$ ./darknet detector demo data/obj.data cfg/yolov4-obj.cfg yolov4-obj_6577.weights -dont_show /YOLOv4-object/data/kitchen.mp4 -i 0 -thresh 0.3 -/YOLOv4-object/data/prediction.avi 
```

**test FPS**
```sh
$ ./darknet detector demo data/obj.data cfg/yolov4-obj.cfg yolov4-obj_6577.weights -dont_show /YOLOv4-object/data/kitchen.mp4 -benchmark
```
