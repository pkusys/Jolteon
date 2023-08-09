# Video-Analytics

This folder contains the source code of the Video-Analytics application.
Application, Video-Analytics, consists of four stages: Split, Extract, Preprocess and Classify

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

4. [Test](LGB-Code/): Code for **Test** function. This stage aggreate the multiple forests into a single forest and test the final model with a held-out test dataset.
```
bash
```