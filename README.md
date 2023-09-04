# Do Not Disturb Me: Person Re-identification Under the Interference of Other Pedestrians (ECCV 2020)

Official code for ECCV 2020 paper [Do Not Disturb Me: Person Re-identification Under the Interference of Other Pedestrians](https://arxiv.org/abs/2008.06963).

<p align="center">
	<img src=image/examples.png width=80% />
<p align="center">

## Introduction

In the conventional person Re-ID setting, it is assumed that cropped images are the person images within the bounding box for each individual. However, in a crowded scene, off-shelf-detectors may generate bounding boxes involving multiple people, where the large proportion of background pedestrians or human occlusion exists. The representa- tion extracted from such cropped images, which contain both the target and the interference pedestrians, might include distractive information. This will lead to wrong retrieval results. To address this problem, this paper presents a novel deep network termed Pedestrian-Interference Sup- pression Network (PISNet). PISNet leverages a Query-Guided Attention Block (QGAB) to enhance the feature of the target in the gallery, under the guidance of the query. Furthermore, the involving Guidance Reversed Attention Module and the Multi-Person Separation Loss promote QGAB to suppress the interference of other pedestrians. Our method is evalu- ated on two new pedestrian-interference datasets and the results show that the proposed method performs favorably against existing Re-ID methods.


<h2 id="jump">Resouces</h2>

1. Pretrained Models:

   [Baidu NetDisk](https://pan.baidu.com/s/1O08TssJcASsTh8veIBimzA), Password: 6x4x. The Models are trained using the gt boxes from [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and [PRW](https://github.com/liangzheng06/PRW-baseline), respectively.
   
2. Datasets：
   
   Request the datasets from xbrainzsz@gmail.com (academic only).
   Due to licensing issues, please send me your request using your university email.

## Citation

If you find this code useful in your research, please consider citing:
```
@inproceedings{zhao2020pireid,
  title={Do Not Disturb Me: Person Re-identification Under the Interference of Other Pedestrians},
  author={Shizhen, Zhao and Changxin, Gao and Jun, Zhang and Hao, Cheng and Chuchu, Han and Xinyang, Jiang and Xiaowei, Guo and Wei-Shi, Zheng and Nong, Sang and Xing, Sun},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## Contact

Shizhen Zhao: xbrainzsz@gmail.com
