# AI-2Dstyle-Transfer
基于DCT-NET风格迁移算法的视频自动化风格修改demo

> [**DCT-Net: Domain-Calibrated Translation for Portrait Stylization**](arxiv_url_coming_soon),             
> [Yifang Men](https://menyifang.github.io/)<sup>1</sup>, Yuan Yao<sup>1</sup>, Miaomiao Cui<sup>1</sup>, [Zhouhui Lian](https://www.icst.pku.edu.cn/zlian/)<sup>2</sup>, Xuansong Xie<sup>1</sup>,        
> _<sup>1</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com), Beijing, China_  
> _<sup>2</sup>[Wangxuan Institute of Computer Technology, Peking University](https://www.icst.pku.edu.cn/), China_     
> In: SIGGRAPH 2022 (**TOG**) 
> *[arXiv preprint](https://arxiv.org/abs/2207.02426)* 

<a href="https://colab.research.google.com/github/menyifang/DCT-Net/blob/main/notebooks/inference.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SIGGRAPH2022/DCT-Net)


## Demo
![demo_vid](https://github.com/menyifang/DCT-Net/raw/main/assets/demo.gif)


## Requirements
* python >= 3.7
* tensorflow >=1.14
* CuDNN == 11.3.1
* CUDA  == 8.1.0
* easydict
* numpy
* both CPU/GPU are supported


## Quick Start

- 下载并安装ModelScope library

```bash
conda create -n dctnet python=3.8
conda activate dctnet
conda install tensorflow==2.10
conda install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

- 模型加载和推理demo
```bash
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-portrait-stylization', 'damo/cv_unet_person-image-cartoon_compound-models')
```

- 运行视频转绘画风格demo
```bash
python demo.py
```

A quick use with python SDK

- Downloads:
```bash
python download.py
```

- Inference:
```bash
python run_sdk.py
```


### From source code
```bash
python run.py
```

## Multi-style

Multi-style models and usages are provided here.

![demo_img](https://raw.githubusercontent.com/menyifang/DCT-Net/main/assets/styles.png)

```bash
git clone https://github.com/menyifang/DCT-Net.git
cd DCT-Net
```

###  Multi-style models download

- upgrade modelscope>=0.4.7

```bash
conda activate dctnet
pip install --upgrade "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

- Download the pretrained models with specific styles [option: anime, 3d, handdrawn, sketch, artstyle]
```bash
python multi-style/download.py --style 3d
```

### Inference

- Quick infer with python SDK, style choice [option: anime, 3d, handdrawn, sketch, artstyle]

```bash
python multi-style/run_sdk.py --style 3d
```

- Infer from source code & downloaded models
```bash
python multi-style/run.py --style 3d
```


## Reference

```bibtex
@inproceedings{men2022dct,
  title={DCT-Net: Domain-Calibrated Translation for Portrait Stylization},
  author={Men, Yifang and Yao, Yuan and Cui, Miaomiao and Lian, Zhouhui and Xie, Xuansong},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  pages={1--9},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```
