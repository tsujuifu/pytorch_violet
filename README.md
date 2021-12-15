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
+ [DALL-E](https://github.com/openai/DALL-E)
+ [transformers](https://pypi.org/project/transformers/)

## Usage
### Data preprocessing
As using outer datasets, we provide [preprocessing tools](https://github.com/tsujuifu/pytorch_violet/tree/main/_tools) to extract **sparse-sampled video frames** into our compressed format.
```
cd _tools

# We use 4 frames during pretraining and 5 frames for downstream tasks
python extract_video-frame.py --path=msrvtt --sample=5 # msrvtt.pkl

# We use DALL-E to extract VQ tokens for MVM pretraining
python extract_vq.py --path=msrvtt --frame=224 # msrvtt_vq.pkl

# We adopt file.seek() instead of loading entire data to reduce the memory cost during distributed pretraining
python extract_tsv.py --path=msrvtt # msrvtt.tsv, msrvtt.lineidx
```
There are [examples](https://github.com/tsujuifu/pytorch_violet/tree/main/_data) (webvid2.5m, cc3m, tgif-action, msvd-qa, and msrvtt-retrieval) to help formulate the input data.

### Pretraining
Put [pretrained VT](https://drive.google.com/file/d/1B1tkA9EnlQlK72xB8liz_WRo7WTEpJDt/view?usp=sharing) in [\_snapshot](https://github.com/tsujuifu/pytorch_violet/tree/main/_snapshot).
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 --master_port=7122 main_pretrain.py
```

## Citation
```
@inproceedings{fu2021violet, 
  author = {Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu}, 
  title = {VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling}, 
  booktitle = {arXiv:2111.1268}, 
  year = {2021} 
}
```
