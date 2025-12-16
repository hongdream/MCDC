# Robust Categorical Data Clustering Guided by Multi-Granular Competitive Learning

This file is our implementation of "Robust Categorical Data Clustering Guided by Multi-Granular Competitive Learning".
Contact:

- tauhongcai@gmail.com (Note that the e-mail shown in our paper is unavailable now.)

![image-20251216153128194](C:\Users\33705\AppData\Roaming\Typora\typora-user-images\image-20251216153128194.png)

## How to Run MCDC

Just run "MCDC_demo.m", the the experimental results will be displayed automatically.

```matlab
file = "./Dataset/xxx";
```

xxx: select the dataset you want to run

## File description

All the folders and files for implementing the proposed MCDC algorithm are introduced below:

- Dataset: A folder contains public/benchmark datasets used in the corresponding paper.
  - MCDC_demo.m: A script to cluster different datasets in the Dataset folder using the proposed method.
  - MGCPL.m: A function implements the cluster distributions exploration at multi-granular levels.
  - GAME.m: A function that implements cluster assignment adaptively using the multi-granular cluster distributions.
  - class_assign.m: A function that calculates the object-cluster similarity. 
  - OI.m: A function that implements oriented initialization for categorical data clustering
  - Weight.m: A function that calculates the weight of each attribute.

## Citation

```tex
@inproceedings{cai2024robust,
  title={Robust categorical data clustering guided by multi-granular competitive learning},
  author={Cai, Shenghong and Zhang, Yiqun and Luo, Xiaopeng and Cheung, Yiu-Ming and Jia, Hong and Liu, Peng},
  booktitle={2024 IEEE 44th International Conference on Distributed Computing Systems (ICDCS)},
  pages={288--299},
  year={2024},
  organization={IEEE}
}
```
