# Enhancing Charcoal Species Identification through Optimized Transfer Learning and Saliency Analysis

Course: Advanced Topics in Data Science: Deep Learning (DS-UA 301)

Team Members: Ryan Li, Hana Liu, alex dong


## Objective

The goal of this project is to develop a Vision Transformer (ViT) model capable to identifying charcoal species from microscopic images. This model aims to assist in the analysis of fossilized charcoal to better understand early human settlements and their environmental interactions. Additionally the project will explore the impact of pre-training dataset size and similarity on the effectiveness of transfer learning.


## Background and Significance

Fire holds significance as both a natural element and a cultural artifact. It's presence in paleo-environmental records is predominantly recorded in the form of fossilized charcoal. These remnants are critical for the study of human history and ecological shifts but are challenging to analyze due to the specialized knowledge and training required. Machine learning techniques, particularly transfer learning, offer a promising avenue to simplify and expedite this analysis.

Transfer learning is a technique where a model trained for a particular task is reused as the starting point for a model on a second task. This technique has benefits in efficiency, performance, and flexibility. Computational cost can be reduced by using pre-existing models, performance can improve in cases where not much data is readily available, and new models can be developed by simply tweaking older models. The most computationally expensive and time consuming part of transfer learning usually comes during pre-training, as models need to have been trained on a sufficiently vast and compresensive dataset to learn general features. 


## Project Description

### Machine Learning Model Development

- **Model Choice**: This project will employ a Vision Transformer (ViT), a model architecture which has shown substantial success in image classification tasks. ViT models process images in a manner similar to how transformers handle sequences in NLP, making them suitable for nuanced image recognition tasks such as distinguishing between charcoal species.

- **Image Data**: Microscopic TIF images of various charcoal species will serve as the primary data for fine-tuning and testing the model. 

### Transfer Learning Strategy 

- **Pre-training Exploration**: Before fine-tuning on charcoal images, the ViT will undergo pre-training. Two scenarios will be compared:
    - Pre-training on ImageNet, a large and diverse dataset common for initial training of image recognition models.
    - Pre-training on a smaller dataset of cells, which are more closely related to the charcoal images in terms of visual and contextual features.

- **Hypothesis**: A smaller, more similar dataset might yield comparable or superior performance to ImageNet pre-training due to better initial feature alignment with the end task.

### Saliency Map Integration for Feature Analysis

- **Saliency Map Generation**: Implement gradient-based methods to create saliency maps. These maps will visualize the most influential regions and features in the images as percieved by the model, providing an indication of how the model is making its decisions.

- **Pre and Post Fine-tuning Analysis**: Generate and compare saliency maps from models pre-trained on both datasets before and after fine-tuning on the charcoal dataset. This comparison will visually demonstrate how dataset characteristics influence feature prioritization and learning.

- ~~**Quantitative Evaluation**: Use metrics such as Intersection over Union (IoU) to quantify the overlap and consistency of salient features, aiding in a robust comparison between different training strategies.~~

### Evaluation Metrics

- **Performance Indicators**: Precision, Recall, and F1-Score across various classes of charcoal will be measured to evaluate each model's effectiveness.

- **Visual and Assessments**: Use saliency maps for qualitative analysis and overlap metrics for quantitative validation.


## Repo Overview
.
├── README.md
├── data
│   ├── NumberOfFilesBBBC022.txt
│   ├── charcoal-images.txt
│   └── pretraining-data-info.txt
├── model-checkpoints
│   ├── info.txt
│   └── load_ckpt_example.ipynb
├── notebooks
│   ├── finetuning.ipynb
│   ├── pretraining-subset.ipynb
│   ├── pretraining-test.ipynb
│   └── pretraining.ipynb
└── scripts
    ├── dataset_indices.py
    ├── merge_classes.py
    ├── pretraining-test.py
    ├── pretraining.py
    ├── progress_bar.py
    ├── run-pretraining-test.sbatch
    └── run-pretraining.sbatch

The organization of the repo is relatively straightforward. Information about the data files is in data/. Information about the model checkpoints are in model-checkpoints/. Unfortunately these directories do not contain the actual data or model checkpoints due to size constraints. The scripts/ directory contains python scripts of the notebooks along with some utility scripts to make life easier. 
