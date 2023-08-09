# Video-Analytics

This folder contains the source code of the Video-Analytics application.
Application, Video-Analytics, consists of four stages: Split, Extract, Preprocess and Classify

## Folder Contents
1. [Split](src/split.py): Code for **Split** function. This stage preprocesses the video data, which split the videos into chunks. To deploy *Split*, run:
```
bash
```

2. [Extract](src/extract.py): Code for **Train** function. This stage trains a user-specified number of decision tree model with LightGBM. A lambda function consists of a number of processes. Each process is responsible for training a tree model.
To deploy *Train*, run: 
```
bash
```

3. [Aggregation](LGB-Code/): Code for **Aggregate** function. This stage aggreate the **Train** stage model and merge them into a forest. (You can use multiple functions here and create multiple forests. The next stage is responsible to merge the multiple forests into one forest, called parallel merging).
```
bash
```

4. [Test](LGB-Code/): Code for **Test** function. This stage aggreate the multiple forests into a single forest and test the final model with a held-out test dataset.
```
bash
```