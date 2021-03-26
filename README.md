# PU-PieAPP - Deep Photometric Image Quality Metric

The repository contains the code for the Deep High-Dynamic-Range Image Quality metric -- PU-PieAPP trained on the [Unified Photometric Image Quality (UPIQ) dataset](https://www.repository.cam.ac.uk/handle/1810/315373)

## Description

Increasing popularity of HDR image and video content brings the need for metrics that could predict the severity of image impairments as seen on displays of different brightness levels and dynamic range. Such metrics should be trained and validated on a sufficiently large subjective image quality dataset to ensure robust performance. As the existing HDR quality datasets are limited in size, we created a Unified Photometric Image Quality dataset (UPIQ) with over 4,000 images by realigning and merging existing HDR and SDR datasets. The realigned quality scores share the same unified quality scale across all datasets. Such realignment was achieved by collecting additional cross-dataset quality comparisons and re-scaling data with a psychometric scaling method. Images in the proposed dataset are represented in absolute photometric and colorimetric units, corresponding to light emitted from a display. We use the new dataset to train a new deep photometric image quality metric (PU-PieAPP) which outperforms existing metrics on high-dynamic image quality prediction. For more information please refer to the [project page](https://www.cl.cam.ac.uk/research/rainbow/projects/upiq/).

## Usage

The code runs in Python3 with Pytorch. First install the required dependencies:

```
pip3 install -r requirements.txt
```

An example running the metric:



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
```

## Acknowledgements

This project has received funding from EPSRC research grants EP/P007902/1 and EP/R013616/1, from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement N725253 (EyeCode), and from the Marie Skłodowska-Curie grant agreement N765911 (RealVision).
