# FlawMatch
Official Repository for "FlawMatch: Conditional Defect Image Generation via Flow Matching for Improved Surface Defect Classification"
This repository contains the data and code used for the experiments in our paper, which are based on the public dataset KolektorSDD2.

Before running the code, please install the required packages listed in `requirements.txt`.
All experiments were conducted with Python 3.8.

## Dataset
Our work does not directly use KolektorSDD2 as input. Instead, we crop the defect regions and use them for training and evaluation.
The related dataset can be downloaded via the following link. 
[https://drive.google.com/drive/folders/1bfhsZqW4Ojjfk3dMwnZjhniuNhJ_A1Ng?usp=sharing]


## Code

The code used for training is `train_1c_with_uncondition.py`, and its usage is described below.
```
python train_1c_with_uncondition.py --batch_size 1024 --n_eps 50  --lr 1e-4  --label_num 233 --wh_w 64  --wh_h 64  --layer 5 --resblock 1    --dataset path_to_traindata
```

The code used for inference is `sampling_1c_cond.py`, and its usage is described below.
```
python sampling_1c_cond.py --layer 5 --wh_w 64 --wh_h 64 --store_path path_to_model_pth
```

The `torchcfm` folder is originally based on the code from https://github.com/atong01/conditional-flow-matching.
We would like to thank **Alexander Tong** for creating and sharing the related code.


## Citation
If you find our work helpful, please consider citing the following paper and ⭐ the repo.

```
@article{OH2025103704,
  author = {Hyunwoo Oh and Seunghee Choi and Jinho Baek and Dong-Jin Kim and Junegak Joung},
  title = {FlawMatch: Conditional defect image generation via flow matching for improved surface defect classification},
  journal = {Advanced Engineering Informatics},
  volume = {68},
  pages = {103704},
  year = {2025},
  doi = {https://doi.org/10.1016/j.aei.2025.103704}
}
```
