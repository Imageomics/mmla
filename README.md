# MMLA Repo
Multi-Environment, Multi-Species, Low-Altitude Aerial Footage Dataset

![zebras_giraffes](vizualizations/location_1_session_5_DJI_0211_partition_1_DJI_0211_002590.jpg)
Example photo from the MMLA dataset and labels generated from model. The image shows a group of zebras and giraffes at the Mpala Research Centre in Kenya. 
## Table of Contents
  - [How to use the scripts in this repo](#how-to-use-the-scripts-in-this-repo)
    - [Requirements](#requirements)
  - [Baseline YOLO evaluation](#baseline-yolo-evaluation)
    - [Download evaluation data from HuggingFace](#download-evaluation-data-from-huggingface)
    - [Run the evaluate_yolo script](#run-the-evaluate_yolo-script)
  - [Model Training](#model-training)
    - [Prepare the dataset](#prepare-the-dataset)
    - [Optional: Downsample the frames](#optional-downsample-the-frames)
    - [Run the training script](#run-the-training-script)
  - [Evaluation](#evaluation)
    - [Optional: Perform bootstrapping](#optional-perform-bootstrapping)
  - [Results](#results)
  - [Fine-Tuned Model Weights](#fine-tuned-model)
  - [Paper](#paper)
  - [Dataset](#dataset)

This repo provides scripts to fine-tune YOLO models on the MMLA dataset. The [MMLA dataset](https://huggingface.co/collections/imageomics/wildwing-67f572d3ba17fca922c80182) is a collection of low-altitude aerial footage of various species in different environments. The dataset is designed to help researchers and practitioners develop and evaluate object detection models for wildlife monitoring and conservation.


# How to use the scripts in this repo

### Requirements
```bash
# install packages from requirements
conda create --name yolo_env --file requirements.txt
# OR using pip
pip install -r requirements.txt
```



## Baseline YOLO evaluation
### Download evaluation data from HuggingFace
This dataset contains an evenly distributed set of frames from the MMLA dataset, with bounding box annotations for each frame. The dataset is designed to help researchers and practitioners evaluate the performance of object detection models on low-altitude aerial footage containing a variety of environments and species.

```bash
# download the datasets from HuggingFace to local /data directory 

git clone 
```

### Run the evaluate_yolo script
```bash
# example usage
python model_eval/evaluate_yolo.py --model model_eval/yolov5mu.pt  --images model_eval/eval_data/frames_500_coco --annotations model_eval/eval_data/frames_500_coco --output model_eval/results/frames_500_coco/yolov5m

```
## Model Training

### Prepare the dataset
```bash
# download the datasets from HuggingFace to local /data directory 

# wilds dataset
git clone https://huggingface.co/datasets/imageomics/wildwing_wilds
# opc dataset
git clone https://huggingface.co/datasets/imageomics/wildwing_opc
# mpala dataset
git clone https://huggingface.co/datasets/imageomics/wildwing_mpala

# run the script to split the dataset into train and test sets
python prepare_yolo_dataset.py

```

#### Alternatively, you can create your own dataset from video frames and bounding box annotations
```bash
python frame_extractor.py --dataset wilds --dataset_path ./wildwing_wilds --output_dir ./wildwing_wilds

```
### Optional: Downsample the frames to extract a subset of frames from each video
```bash
python downsample.py --dataset wilds --dataset_path ./wildwing_wilds --output_dir ./wildwing_wilds --downsample_rate 0.1
```

### Run the training script
```bash
# run the training script
python train.py
```

## Evaluation
To evaluate the trained model on the test data:
```bash
# run the validate script
python validate.py 
```

### Optional: Perform bootstrapping to get confidence intervals
```bash
# run the evaluation script
bootstrap.ipynb
```
#### Download inference results from baseline and fine-tned model 

## Results
Our fine-tuned YOLO11m model achieves the following performance on the MMLA dataset:
| Class   | Images | Instances | Box(P) | R     | mAP50 | mAP50-95 |
|---------|--------|-----------|--------|-------|-------|----------|
| all     | 7,658  | 44,619    | 0.867  | 0.764 | 0.801 | 0.488    |
| Zebra   | 4,430  | 28,219    | 0.768  | 0.647 | 0.675 | 0.273    |
| Giraffe | 868    | 1,357     | 0.788  | 0.634 | 0.678 | 0.314    |
| Onager  | 172    | 1,584     | 0.939  | 0.776 | 0.857 | 0.505    |
| Dog     | 3,022  | 13,459    | 0.973  | 0.998 | 0.995 | 0.860    |


# Fine-Tuned Model
See [HuggingFace Model Repo](https://huggingface.co/imageomics/mmla) for details and weights.

# Dataset
See [HuggingFace Dataset Repo](https://huggingface.co/collections/imageomics/wildwing-67f572d3ba17fca922c80182) for MMLA dataset.

# Paper
```bibtex
@article{kline2025mmla,
  title={MMLA: Multi-Environment, Multi-Species, Low-Altitude Aerial Footage Dataset},
  author={Kline, Jenna and Stevens, Samuel and Maalouf, Guy and Saint-Jean, Camille Rondeau and Ngoc, Dat Nguyen and Mirmehdi, Majid and Guerin, David and Burghardt, Tilo and Pastucha, Elzbieta and Costelloe, Blair and others},
  journal={arXiv preprint arXiv:2504.07744},
  year={2025}
}
```