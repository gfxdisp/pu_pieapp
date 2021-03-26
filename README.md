# Consolidated Dataset and Metrics for High-Dynamic-Range Image Quality

Increasing popularity of HDR image and video content brings the need for metrics that could predict the severity of image impairments as seen on displays of different brightness levels and dynamic range. Such metrics should be trained and validated on a sufficiently large subjective image quality dataset to ensure robust performance. As the existing HDR quality datasets are limited in size, we created a Unified Photometric Image Quality dataset (UPIQ) with over 4,000 images by realigning and merging existing HDR and SDR datasets. The realigned quality scores share the same unified quality scale across all datasets. Such realignment was achieved by collecting additional cross-dataset quality comparisons and re-scaling data with a psychometric scaling method. Images in the proposed dataset are represented in absolute photometric and colorimetric units, corresponding to light emitted from a display. We use the new dataset to retrain existing HDR metrics and show that the dataset is sufficiently large for training deep architectures. We show the utility of the dataset on brightness aware image compression.


## Usage

The code runs in Python3 with Pytorch. First install the required dependencies:

```
pip3 install -r requirements.txt

```

## Citing


```
@misc{Mikhailiuk2021_1, 
author = {Mikhailiuk, Aliaksei and Perez-Ortiz, Maria and Yue, Dingcheng and Suen, Wilson and Mantiuk, Rafa{\l} K.}, 
eprint={2012.10758}, 
archivePrefix={arXiv}, 
primaryClass={eess.IV}, 
title = {{Consolidated Dataset and Metrics for High-Dynamic-Range Image Quality}}, 
year = {2020} 
}


