# Video-Analytics

This folder contains the source code of the Video-Analytics application.
Application, Video-Analytics, consists of four stages: Split, Extract, Preprocess and Classify.
Get the yolo model through [imageai-doc](https://imageai.readthedocs.io/en/latest/video/index.html?highlight=yolo-tiny#video-and-live-feed-detection-and-analysis) in this folder before depolying.

## Folder Contents
1. [Split](src/split.py): Code for **Split** function. This stage preprocesses the video data, which split the videos into chunks. To deploy *Split*, run:
```
bash depoly_split.sh
```

2. [Extract](src/extract.py): Code for **Extract** function. This stage extracts a representative frame of each chunk of video in **Split** stage. The frame is sent to either **Preprocess** stage or **Classify** stage.
To deploy *Extract*, run: 
```
bash depoly_extract.sh
```

3. [Preprocess](src/preprocess.py): Code for **Preprocess** function. This stage applies a sharpening filter to improve image quality (only for a half of images).
```
bash deploy_preprocess.sh
```

4. [Test](src/classify.py): Code for **Classify** function. This stage YOLO model to detect the feature
in each frame (1000 classes) and output the classification results in json file.
```
bash deploy_classify.sh
```