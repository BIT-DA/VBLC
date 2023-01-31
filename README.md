# VBLC for Domain Adaptive Semantic Segmentation under Adverse Conditions

[VBLC: Visibility Boosting and Logit-Constraint Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](https://arxiv.org/abs/2211.12256). **AAAI 2023 Oral**.

**by [Mingjia Li](https://kiwixr.github.io), [Binhui Xie](https://binhuixie.github.io), [Shuang Li](https://shuangli.xyz), [Chi Harold Liu](https://scholar.google.com/citations?user=3IgFTEkAAAAJ&hl=en) and [Xinjing Cheng](https://scholar.google.com/citations?user=8QbRVCsAAAAJ&hl=en)**

<!-- **[[arXiv](https://arxiv.org/abs/2211.12256)]** -->
[![Project Page](https://img.shields.io/badge/Project%20Page-%23D80082?logo=&style=flat-square)](https://kiwixr.github.io/projects/vblc)&nbsp;&nbsp;
[![arXiv](https://img.shields.io/badge/Paper-arXiv-%23B31B1B?style=flat-square)](https://arxiv.org/abs/2211.12256)&nbsp;&nbsp;
[![Bilibili](https://img.shields.io/badge/Video-Bilibili-%2300A1D6?logo=bilibili&style=flat-square)](https://www.bilibili.com/video/av692744964)&nbsp;&nbsp;
[![YouTube](https://img.shields.io/badge/Video-YouTube-%23FF0000?logo=youtube&style=flat-square)](https://www.youtube.com/watch?v=o0voyaQ3FzM)&nbsp;&nbsp;
[![Dropbox](https://img.shields.io/badge/Poster-Dropbox-%230061FF?logo=dropbox&style=flat-square)](https://www.dropbox.com/s/pvb2701k2gr9cfb/aaai23poster.pdf?dl=0)&nbsp;&nbsp;

> Update on 2023/01/31: Project page available.
>
> Update on 2023/01/16: We are happy to announce that VBLC was accepted for **Oral Presentation** at AAAI 2023. 🎉🎉🎉
>
> Update on 2023/01/12: Video and poster available.

<!-- TOC -->

- [Overview](#overview)
- [Installation](#installation)
- [Datasets Preparation](#datasets-preparation)
  - [Download Datasets](#download-datasets)
  - [Setup Datasets](#setup-datasets)
- [Model Zoo](#model-zoo)
  - [Task 1: Cityscapes → ACDC](#task-1-cityscapes--acdc)
  - [Task 2: Cityscapes → FoggyCityscapes + RainCityscapes](#task-2-cityscapes--foggycityscapes--raincityscapes)
- [VBLC Evaluation](#vblc-evaluation)
- [VBLC Training](#vblc-training)
- [How to Add VBLC to your Self-Training UDA Code](#how-to-add-vblc-to-your-self-training-uda-code)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

<!-- /TOC -->

## Overview

We propose **VBLC** to tackle the problem of domain adaptive semantic segmentation under adverse conditions. VBLC explores the potential of *getting rid of reference images* and *resolving the mixture of adverse conditions* simultaneously.

The ***visibility boost module*** is introduced in the input space to narrow the gap brought about by poor visibility, while the ***logit constraint learning*** loss is devised to relieve the pain of overconfidence common in self-training schemes.

Extensive analysis and visualizations can be found in our [paper](https://arxiv.org/abs/2211.12256).

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>

## Installation

This code is implemented with `Python 3.8.5` and `PyTorch 1.10.2` on `CUDA 11.3`.

To try out this project, it is recommended to set up a virtual environment first:

```bash
# create and activate the environment
conda create --name vblc -y python=3.8.5
conda activate vblc

# install the right pip and dependencies for the fresh python
conda install -y ipython pip
```

Then, the dependencies can be installed by:

```bash
# install required packages
pip install -r requirements.txt

# install mmcv-full, this command compiles mmcv locally and may take some time
pip install mmcv-full==1.5.0  # requires other packeges to be installed first
```

**Alternatively**, the `mmcv-full` package can be installed faster with official pre-built packages, for instance:

```bash
# another way to install mmcv-full, faster
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

The environment is now fully prepared.

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>

## Datasets Preparation

### Download Datasets

- **Cityscapes:** Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to a custom directory.
- **ACDC:** Download rgb_anon_trainvaltest.zip and gt_trainval.zip from [here](https://acdc.vision.ee.ethz.ch/download/) and extract them to a custom directory.
- **FoggyCityscapes:** Download leftImg8bit_trainvaltest_foggy.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to ***the same directory as Cityscapes dataset***.
- **RainCityscapes:** Download leftImg8bit_trainval_rain.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to ***the same directory as Cityscapes dataset***.

### Setup Datasets

Symlink the required datasets:

```bash
ln -s /path/to/cityscapes/dataset data/cityscapes
ln -s /path/to/acdc/dataset data/acdc
```

Perform preprocessing to convert label IDs to the train IDs and gather dataset statistics:

```bash
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

Ultimately, the data structure should look like this:

```shell
VBLC
├── ...
├── data
│   ├── acdc
│   │   ├── gt
│   │   └── rgb_anon
│   ├── cityscapes
│   │   ├── gtFine
│   │   ├── leftImg8bit
│   │   ├── leftImg8bit_foggy
│   │   ├── leftImg8bit_rain
├── ...
```

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>

## Model Zoo

We provide pretrained models of both Domain Adaptive Semantic Segmentation tasks through [Google Drive](https://drive.google.com/drive/folders/1lqnUBRSAF-fKIUI2VsO3tYgRAa43eurN?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1TXA0egGCJ6iA44mSrXHAMQ) (access code: `VBLC`).

### Task 1: Cityscapes &rarr; ACDC

| model name                  | backbone   | mIoU (test) | mIoU (val) | checkpoint download |
| :-------------------------- | :--------- | :---------- | :--------- | :------------------ |
| vblc_city2acdc_dlv2.pth     | ResNet-101 | 47.8        | 46.0       | [Google](https://drive.google.com/drive/folders/1EZf-UWqgoWTst4OkKXAgdDAgVKih6Yit?usp=sharing) / [Baidu](https://pan.baidu.com/s/1LxapqMSYcU7L9DnRLNWOqg) (acc: `VBLC`) |
| vblc_city2acdc_daformer.pth | MiT-B5     | 64.2        | 63.7       | [Google](https://drive.google.com/drive/folders/1rbUpKYMGbDtf2XjY_sL_F8tOVOy3s0vo?usp=sharing) / [Baidu](https://pan.baidu.com/s/1UjY2jw4uK2uu15Kxu0I8LA) (acc: `VBLC`) |

### Task 2: Cityscapes &rarr; FoggyCityscapes + RainCityscapes

| model name                       | backbone   | mIoU (val) | checkpoint download |
| :------------------------------- | :--------- | :--------- | :------------------ |
| vblc_city2foggyrain_dlv2.pth     | ResNet-101 | 74.9       | [Google](https://drive.google.com/drive/folders/1iWbQWWjwoMVeejDpBjk6jTSPzi4dr19_?usp=sharing) / [Baidu](https://pan.baidu.com/s/1YpTyG15AtbgQZPB2iVnF2A) (acc: `VBLC`) |
| vblc_city2foggyrain_daformer.pth | MiT-B5     | 79.8       | [Google](https://drive.google.com/drive/folders/1wXMGYr2Qvzgvkxt1W3PdZC8MYX4pPBVy?usp=sharing) / [Baidu](https://pan.baidu.com/s/1K7rLAUsT5_-jDNkU3Kx-VQ) (acc: `VBLC`) |

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>

## VBLC Evaluation

To evaluate the pretrained models on the **validation set**, please run as follows:

```bash
python -m tools.test /path/to/config /path/to/checkpoint --eval mIoU --val
```

<details>
<summary>Example</summary>

For example, if you download `vblc_city2acdc_daformer.pth` along with its config json file `vblc_city2acdc_daformer.json` into folder `./checkpoints/vblc_city2acdc_daformer/`, then the evaluation script should be like:

```bash
python -m tools.test ./checkpoints/vblc_city2acdc_daformer/vblc_city2acdc_daformer.json ./checkpoints/vblc_city2acdc_daformer/vblc_city2acdc_daformer.pth --eval mIoU --val
```

</details>

Additionally, for ACDC Dataset, you can also obtain the test results through:

```bash
python -m tools.test /path/to/config /path/to/checkpoint --format-only --eval-options imgfile_prefix=/path/prefix/labelTrainIds do_palette=False
```

<details>
<summary>Example</summary>

For the aforementioned example, the corresponding script is:

```bash
python -m tools.test ./checkpoints/vblc_city2acdc_daformer/vblc_city2acdc_daformer.json ./checkpoints/vblc_city2acdc_daformer/vblc_city2acdc_daformer.pth --format-only --eval-options imgfile_prefix=./test_results/labelTrainIds do_palette=False
```

</details>

The test results should be under folder `/path/prefix/labelTrainIds`, and can then be zipped and submitted to [ACDC test server](https://acdc.vision.ee.ethz.ch/submit) for the test score.

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>


## VBLC Training

To begin with, download [SegFormer](https://github.com/NVlabs/SegFormer)'s official MiT-B5 weights (i.e., `mit_b5.pth`) pretrained on ImageNet-1k from [here](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) and put it into a new folder `./pretrained`.

The training entrance is at `run_experiments.py`. To examine the setting for a specific task, please take a look at `experiments.py` for more details. Generally, the training script is given as:

```bash
python run_experiments.py --exp <exp_id>
```

The mapping between `<exp_id>` and tasks:

| `<exp_id>` | task                                               | backbone   |
| :--------: | :------------------------------------------------- | :--------- |
|    `0`     | Test the setup of the environment                  | All        |
|    `1`     | Cityscapes &rarr; ACDC                             | MiT-B5     |
|    `2`     | Cityscapes &rarr; ACDC                             | ResNet-101 |
|    `3`     | Cityscapes &rarr; FoggyCityscapes + RainCityscapes | MiT-B5     |
|    `4`     | Cityscapes &rarr; FoggyCityscapes + RainCityscapes | ResNet-101 |

After training, the models can be validated and tested following [VBLC Evaluation](#vblc-evaluation). Note that the training results are located in `./work_dirs`. The config filename should look like: `220824_2303_city2acdc_vblc_rcs001_daformer_sepaspp_logit_constraint_mitb5_poly10warm_s0_4b56d.json`, and the model file has suffix `.pth`.

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>

## How to Add VBLC to your Self-Training UDA Code

- To include the *visibility boost module*, take a look at `night_fog_filter` method in `mmseg/models/utils/night_fog_filter.py`.
- To include the *logit constraint learning* loss, take a look at `mmseg/models/losses/logit_constraint_loss.py` for implementation details.

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>

## Acknowledgments

This project is based on the following open-source projects. We thank their authors for making the source code publicly available.

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (Apache License 2.0, [license](resources/license_mmseg))
- [SegFormer](https://github.com/NVlabs/SegFormer) (NVIDIA Source Code License, [license](resources/license_segformer))
- [DAFormer](https://github.com/lhoyer/DAFormer) (Apache License 2.0, [license](resources/license_daformer))
- [DACS](https://github.com/vikolss/DACS) (MIT License, [license](resources/license_dacs))
- [IA-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) (Apache License 2.0, [license](resources/license_iayolo))

:star: Also see our [SePiCo](https://github.com/BIT-DA/SePiCo) :star:

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>

## Citation

If you find this work helpful to your research, please consider citing the paper:

```bibtex
@inproceedings{li2023vblc,
  title={VBLC: Visibility Boosting and Logit-Constraint Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions},
  author={Li, Mingjia and Xie, Binhui and Li, Shuang and Liu, Chi Harold and Cheng, Xinjing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={},
  year={2023}
}
```

<div align="right">
<b><a href="#vblc-for-domain-adaptive-semantic-segmentation-under-adverse-conditions">↥</a></b>
</div>