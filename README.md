# VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling
A **PyTorch** implementation of [VIOLET](https://arxiv.org/pdf/2111.12681.pdf)

<img src='_imgs/intro.png' width='50%' />

## Overview
VIOLET is an implementation of <br>
"[VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling](https://arxiv.org/pdf/2111.12681.pdf)" <br>
Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu

<img src='_imgs/violet.png' width='80%' />

VIOLET contains 3 components: **Video Swin Transformer (VT)** computes video features; **Language Embedder (LE)** extracts word embeddings; **Cross-modal Transformer (CT)** performs cross-modal fusion. To benefit from large-scale data, we incorporate 3 pretraining tasks: **Masked Language Modeling (MVM)** predicts the masked word tokens; **Masked Visual-token Modeling (MVM)** recovers the masked video patches; **Visual-Text Matching (VTM)** learns the alignments between video and text modality.

## Requirements
This code is implemented under **Python 3.8**, [PyTorch 1.7](https://pypi.org/project/torch/1.7.0/), and [Torchvision 0.8](https://pypi.org/project/torchvision/0.8.0/). <br>
+ [av](https://pypi.org/project/av/), [tqdm](https://pypi.org/project/tqdm/), [cv2](https://pypi.org/project/opencv-python/)
+ [transformers](https://pypi.org/project/transformers/)

## Usage

## Citation
```
@inproceedings{fu2021violet, 
  author = {Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu}, 
  title = {VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling}, 
  booktitle = {arXiv:2111.1268}, 
  year = {2021} 
}
```
