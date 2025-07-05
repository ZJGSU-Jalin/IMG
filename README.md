
Repo for ACM MM'25 paper "*Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval*". This paper proposes solutions for the Video Moment Retrieval task from an audio-visual collaborative perspective.


![framework](figures/framework.jpeg)


Ubuntu 20.04
CUDA 12.0
Python 3.7


1. Set up the environment

Use Anaconda and easily build up the required environment by

```bash
cd IMG
conda env create -f env.yml
```

2. Data Preparation

Follow previous work [ADPN](https://github.com/hlchen23/ADPN-MM), we use GloVe-840B-300d for text embeddings, I3D visual features and PANNs audio features for Charades-STA dataset, and C3D visual features and VGGish audio features for ActivityNet Captions dataset. Download [data](https://pan.baidu.com/s/1LxdASuOzueq_4YpEr2muAA?pwd=5w4h), touch `IMG/data`, and ensure the following directory structure.

```
|--data
|  |--dataset
|     |--activitynet
|     |     |--train.json
|     |     |--val_1.json
|     |     |--val_2.json
|     |--charades
|     |     |--charades_sta_test.txt
|     |     |--charades_sta_train.txt
|     |     |--charades.json
|     |     |--charades_audiomatter.json
|  |--features
|     |--activitynet
|     |     |--audio
|     |     |     |--VGGish.pickle
|     |     |--c3d_video
|     |     |     |--feature_shapes.json
|     |     |     |--v___c8enCfzqw.npy
|     |     |     |--...(*.npy)
|     |--charades
|     |     |--audio
|     |     |     |--0A8CF.npy
|     |     |     |--...(*.npy)
|     |     |--i3d_video
|     |     |     |--feature_shapes.json
|     |     |     |--0A8CF.npy
|     |     |     |--...(*.npy)
```



3. Training

Train

```bash
python main.py --task <charades|activitynet|charadesAM> --mode train --gpu_idx <GPU INDEX>
```

4. Inference


```bash
python main.py --task <charades|activitynet|charadesAM> --mode test --gpu_idx <GPU INDEX>
```

Change the config `model_name` in `main.py` to the model_name of `your checkpoint`.




## Acknowledgement

We follow the repo [ADPN](https://github.com/hlchen23/ADPN-MM) and [VSLNet](https://github.com/26hzhang/VSLNet) for the code-running framework to quickly implement our work. We appreciate these great jobs.

<!-- ## Cite

If you feel this repo is helpful to your research, please cite our work.

```
@inproceedings{chen2023curriculum,
  title={Curriculum-Listener: Consistency-and Complementarity-Aware Audio-Enhanced Temporal Sentence Grounding},
  author={Chen, Houlun and Wang, Xin and Lan, Xiaohan and Chen, Hong and Duan, Xuguang and Jia, Jia and Zhu, Wenwu},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={3117--3128},
  year={2023}
}
``` -->
