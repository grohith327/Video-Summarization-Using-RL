# Video Summarization using RL
We evaluate the use of augmentations on images in RL, when generating a summary of video. Recent work by [Laskin et al.](https://arxiv.org/pdf/2004.14990.pdf) and [Kostrikov et al.](https://arxiv.org/pdf/2004.13649.pdf) have shown that augmentation on pixels improve the sample efficiency of the RL algorithm and also increases the average cumulative reward obatained. Our work builds on the paper [Deep RL for Unsupervised Video Summarization](https://arxiv.org/pdf/1801.00054.pdf) and evalutes the use of augmentations on pixels/images present in the video to improve the performance of the agent. We also studied the use of [PPO](https://arxiv.org/pdf/1707.06347.pdf) algorithm for training the summary generation agent based on performance (total reward obtained).

Authors: [Rohith](https://github.com/grohith327), [Dibyajit](https://github.com/dibyajit30)

## Run experiments
Use the shell scripts to run each experiment. The shell script runs each experiment on 5 different seeds. The results are stored in a log file. For ex:
```
./train_resnet50_ppo_augs.sh
```

## Augmentations used
- Gaussian Blur
- Cutout
- Cutout Color
- Rotate
- Flip
- Center Crop
- Grayscale

## Dataset and training methodology
We train our model architectures on the [TVSum dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf) using the cross-validation method. Following [Zhou et al.](https://arxiv.org/pdf/1801.00054.pdf), we perform 5 fold cross-validation and report our average out-of-fold F1 score. We also evaluate our model the [SumMe dataset](http://varcity.eu/paper/eccv2014_gygli_vidsum.pdf) to test the generalization of our models. Note that SumMe is an out-of-distribution data and our model is not trained on this dataset.

## Results
We use an Enoder-Decoder model.

