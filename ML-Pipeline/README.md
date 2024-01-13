# ML Pipeline

This folder contains the source code of the ML Pipeline application.
Application, ML-Pipeline, consists of four stages: PCA, Train, Aggregation and Test

## Folder Contents
1. [PCA](PCA/): Code for **Split** function. This stage preprocesses the data with dimensionality 
reduction. To deploy *PCA*, run:
```
cd src
bash deploy_PCA.sh
```

2. [Train](LGB-Code/): Code for **Train** function. This stage trains a user-specified number of decision tree model with LightGBM. A lambda function consists of a number of processes. Each process is responsible for training a tree model.
To deploy *Train*, run: 
```
cd src
bash deploy_Train.sh
```

3. [Aggregation](LGB-Code/): Code for **Aggregate** function. This stage aggreates the **Train** stage model and merge them into a forest. (You can use multiple functions here and create multiple forests. The next stage is responsible to merge the multiple forests into one forest, called parallel merging).
```
cd src
bash deploy_Aggregate.sh
```

4. [Test](LGB-Code/): Code for **Test** function. This stage aggreates the multiple forests into a single forest and test the final model with a held-out test dataset.
```
cd src
bash deploy_Test.sh
```