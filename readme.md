# Description
This projects was made for my Master thesis project.
It enables training and use of models for skeleton-based action recognition.
It uses [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) for generating skeletons.
The projects funcitionalities include:
- Mass pose estimation of videos to create skeleton files.
- Preprocessing skeleton files into dataset packages.
- Data filtering using thresholding and Parametric Pose NMS.
- Skeleton tracking.
- Filling sequences using interpolation, zero-filling, copying neighbors, and Multivariate Imputation by Chained Equation(MICE).
- Training models ST-GCN++ and 2P-GCN.
- Evaluation of them on defined datasets.
- Single file classification with sliding window to enable long video support.

# Project structure
- `/configs` - contains configuration files for the project.
This included AlphaPose detector and estimator configs, as well as the configs for the project.
The configs can be placed anywhere
- `/datasets` - implementation of PyTorches Dataset for loading and serving skeletons during training.
- `/models` - implementation of action classification models.
- `/pose_estimation` - all code for pose estimation. 
This currently includes the AlphaPose pipeline and loader for NTU skeletons.
- `/preprocessing` - functions for filtering, tracking, and filling of skeleton sequences. 
Also includes the feature extraction
- `/procedures` - implementation of functionality that is user-accessible. 
This includes skeleton generation, preprocessing, training, etc.
- `/sample_files` - sample skeletons for tests
- `/shared` - functions and classes shared across the whole projects.

# Sample results
Sample frame result obtained on NTU RGB+D 120 dataset.
<img alt="Type on keyboard action detected on NTU RGB+D 120 dataset" src="/.docs/images/sample.1.png" title="Sample 1"/>

Sample frame result obtained on UT-interaction dataset.
<img alt="Hand shaking action detected on UT-interaction dataset" src="/.docs/images/sample.2.png" title="Sample 2"/>

Preprocessing steps visualized on a sample from NTU RGB+D 120 dataset. 
From top to bottom the sequences have: raw pose estimation results, filtered and tracked sequences, and filled sequences.
<img alt="Preprocessing steps visualized" src="/.docs/images/skeleton_prep.1.png" title="Preprocessing"/>

Graph of results obtained on one of the sequences from UT-interaction dataset.
The graph shows ground truth classes in gray areas together with percentage coverage of them obtained via sliding window aggregation.
<img alt="Sliding window example" src="/.docs/images/sliding_seq2_31.svg" title="Sliding window example"/>
# Model Zoo
Sample models are available on [Google Drive](https://drive.google.com/drive/folders/1_ML8Oka0CmdbqiigL8g9uAnecOcKGP55?usp=drive_link).
All of them are trained on skeletons obtained via pose estimation.
For more details check out the config in the model folder.

| Training Set |   Model  | Eval Accuracy |        Note       |  Link  |
|:------------:|:--------:|:-------------:|:-----------------:|:------:|
|   NTU XSUB   | ST-GCN++ |     91.26%    |                   | [GDrive](https://drive.google.com/drive/folders/1RHSfW6YSYWQsS4oqwA2xLCDi1_222Dh7?usp=drive_link) |
|   NTU XVIEW  | ST-GCN++ |     96.69%    |                   | [GDrive](https://drive.google.com/drive/folders/1zqc5vMti_7HwfdrPnPYb3NnCsRH45oll?usp=drive_link) |
|  NTU120 XSET | ST-GCN++ |     86.69%    |                   | [GDrive](https://drive.google.com/drive/folders/1al-3Xzyr3OzZ5X5F0X3aDprMgozhnA82?usp=drive_link) |
|  NTU120 XSUB | ST-GCN++ |     83.67%    |                   | [GDrive](https://drive.google.com/drive/folders/16OqzPAi7cvHYF5vuPjKNjBmhE7YJb_5-?usp=drive_link) |
|   NTU XSUB   |  2P-GCN  |     97.89%    | Interactions only | [GDrive](https://drive.google.com/drive/folders/1Ex2zeca86pSJ_8QtEXXrNdiSBcsiVFKS?usp=drive_link) |
|   NTU XVIEW  |  2P-GCN  |     99.05%    | Interactions only | [GDrive](https://drive.google.com/drive/folders/1BKHiQDHVoIMXlQXs63LiWOWhjl5WzQl5?usp=drive_link) |
|  NTU120 XSET |  2P-GCN  |     94.03%    | Interactions only | [GDrive](https://drive.google.com/drive/folders/1o54yxjZrDn_0V7qA0GrBRmQZvnBuRpSV?usp=drive_link) |
|  NTU120 XSUB |  2P-GCN  |     92.75%    | Interactions only | [GDrive](https://drive.google.com/drive/folders/1NrN3ZRorzY8VDSjhoIu-I_GyjSyApBF2?usp=drive_link) |


# Installation
The projects uses PyTorch as the library for the neural network models and AlphaPose for the pose estimation.
It was heavily tested using Ubuntu 20.04 and Python version 3.8.
The PyTorch was mostly tested with CUDA support, but it should also run without it.
Installation can be 
1. [Install AlphaPose and its dependencies.](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md)
This will also install PyTorch. Although not specified this library also supports PyTorch 2.
2. Install the rest of dependencies with `python -m pip install -r requirements.txt`

# Commands
Some sample commands that the user can run.

## Generating dataset
Generating AlphaPose skeletons from a folder of video files is the first step to creating a dataset for training.
The following command accepts three paths with positional arguments: config file, input folder, and output folder.
```shell
$ python runner.py generate \
    default_config.yaml \
    ~/nturgb+d_rgb/ \
    ~/Data/ntu_alphapose/
```
This will make the program iterate over all files in the input folder and process them through the pose estimation steps.
The results will be saved as individual files in `SkeletonData` format.

## Preprocessing dataset
For preprocessing the program requires the config file, input folders, and save directory.
```shell
$ python runner.py preprocess \
    default_config.yaml \
    ~/Data/ntu_alphapose/ \
    ~/Data/ntu120_alphapose/ \
    --save-path ~/Data/prepared/
```
This command will create a dataset file or files, depending on the `split_strategy` field in the configuration file.
That is done by iterating over all files and passing them through processing stages as defined in the configuration.
Multiprocessing with a configurable number of processes is used to speed up the computation.
## Training
To train a network, one of the simplest commands is used, as all the training options are defined in the config file.
This includes the number of epochs, the learning rate, the frequency of evaluation on the test set, and many other parameters.
```shell
$ python runner.py train \
    default_config.yaml
```
During training the program logs epoch number, loss, accuracy, and a progress bar for each training and evaluation epoch.
Simplified logs are also written to the file which is defined in the configuration file.
Checkpoints containing network parameters are also saved after every epoch, and when training is done the best checkpoint is selected.

## Evaluation
Evaluation can be done for any model that has been trained.
Optional arguments for this action include the path to the checkpoint file and the path to which the results will be saved.
If the checkpoint file is not provided, the best one saved for the given config is used.
```shell
$ python runner.py evaluate \
    default_config.yaml \
    --model logs/default/best.pth \
    --save-path ~/Data/results/default_results.pkl
```
This will run a single epoch of evaluation on the test set.
It will calculate mean loss and accuracy, which will be printed out.
If the save path is provided the program will write the statistics, as well as full network output for each test sample.

## Classification
The main part of the program is the classification.
Here program requires the config and a video file.
Users can also provide paths for checkpoint file, save file for video, and a save file for window results.
```shell
$ python runner.py classify \
    default_config.yaml \
    ntu_samples/S001C001P001R001A019_rgb.avi \
    --model logs/default/best.pth \
    --save-path ~/Data/results/output.avi \
    --window-save-file ~/Data/results/window_res.pkl
```
The above command will result in the classification pipeline being run as described in the previous chapter.
A progress bar will be shown during classification.
At the end program will write the mean result of all the windows.
The saved video file will have the name of the action at each frame embedded in the top left,
and the output of the classifier for each window will be saved in the window save file.
If a save path is not provided the video will be shown during classification in a pop-up window.

## Visualization
Visualization action allows the user to visualize an already estimated skeleton file.
The program requires a path to the skeleton file and a path to the video on which the skeletons will be displayed.
```shell
$ python runner.py visualize \
    ~/Data/prepared/S001C001P001R001A019.apskel.pkl \
    ntu_samples/S001C001P001R001A019_rgb.avi \
    --save-file ~/Data/results/output.avi
```
The above command will create a video with the visualized skeletons.
If the save-file flag is not used, the video will be previewed to the user in pop-up window.

## Pose Estimation
Estimation enables the user to run AlphaPose on a single video file.
The program expects a path to the configuration file and a path to the video file.
```shell
$ python runner.py estimator \
    default_config.yaml \
    ntu_samples/S001C001P001R001A019_rgb.avi \
    --save-video ~/Data/results/output_video.avi
    --save-skeleton ~/Data/results/output_skeleton.pkl
```
The above instruction will result in the pose estimation being done on the given video.
The results will be saved in the `SkeleontData` format and also applied to the video.
Both save-video and save-skeleton flags are optional.
If the former is absent, the video will be previewed to the user.

# TODO
- [ ] Meaningful unit tests
- [ ] Fix visualization code repetition
- [ ] Test for Windows
- [ ] Add support for HRNet

# Sources
This project uses models obtained from [Pyskl](https://github.com/kennymckormick/pyskl) and pose estimation is done with [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) repositories.
