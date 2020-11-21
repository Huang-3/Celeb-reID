# Celeb-reID

This repository contains Datasets and Code for our paper [Beyond Scalar Neuron: Adopting Vector-Neuron Capsules for Long-Term Person Re-Identification](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8873614) and [Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8851957).

### 1.Dataset
![](https://github.com/Huang-3/Celeb-reID/blob/master/show.jpg)

You can directly download our datasets from OneDrive [Celeb-reID](https://1drv.ms/u/s!Ats-N2hYCkEIgckQF5M5TCsBF0aQZQ?e=IKG16O) and its light version [Celeb-reID-light](https://1drv.ms/u/s!Ats-N2hYCkEIgckRrtd0HGhUZPqNcw?e=ziRTjb).

Baidu Cloud Link:
For Celeb-reID:
Link：https://pan.baidu.com/s/1sKkO4l8FyzM7iXnzjPyWHQ 
code：ix2j 

For Celeb-reID-light:
Link：https://pan.baidu.com/s/13DSJ1PK_AEF9TEGi30eIQA 
code：14k5


The data split of `Celeb-reID` is as follows:

| split  | training|query|gallery|total |
| -------| -----   |-----| ----- | -----| 
| #ID    | 632     |420  |420    |1,052 | 
| #Images| 20,208  |2,972|11,006 |34,186|

The data split of `Celeb-reID-light` is as follows:

| split  | training|query|gallery|total |
| -------| -----   |-----| ----- | -----| 
| #ID    | 490     |100  |100    |590   | 
| #Images| 9021    |887  |934    |10,842|

`Note: The two datasets should be used for research only. Please DO NOT distribute or use it for commercial purpose.`

### 2.Code
![](https://github.com/Huang-3/Celeb-reID/blob/master/show_reidcaps.jpg)
Our code (ReIDCaps) is implemented by Pytorch(>=1.0.0) and python(anaconda 3.6) with Ubuntu.

`To run the training code:`

`First`: Download the Celeb-reID dataset to your own path. Do body part partition by copying the `part_partition.m` file to the path of your dataset. Changing `name=gallery`, `name=query`, and `name=train` to get the body part partition files in the path of your Celeb-reID dataset.

`Second`: Directly run the `run_train.sh` by `bash ./run_train.sh` on console. You may change the path of dataset in the file of `run_train.sh` according to your own path.

`To run the testing code:`

Just simply run the 'test.py' file. You may change the path of dataset and logs(the path of trained model) according to your own path.

You can also directly download our [trained_model](https://1drv.ms/u/s!Ats-N2hYCkEIgckkVxQnNowbn_RNog?e=IEw5Zj) to get the perfomance reported in our paper `Beyond Scalar Neuron: Adopting Vector-Neuron Capsules for Long-Term Person Re-Identification, TCSVT2019`(see Table VII ReIDCaps*(ours)).

| Method          | mAP    |rank-1 |rank-5|
| -------         | -----  |-----  | ---- | 
| ReIDCaps*(Ours) | 15.8%  |63.0%  |76.3% |

### Citation
Please cite this paper in your publications if it helps your research:
```
@article{huang2019beyond,
  title={Beyond Scalar Neuron: Adopting Vector-Neuron Capsules for Long-Term Person Re-Identification},
  author={Huang, Yan and Xu, Jingsong and Wu, Qiang and Zhong, Yi and Zhang, Peng and Zhang, Zhaoxiang},
  journal={Transactions on Circuits and Systems for Video Technology (TCSVT)},
  year={2019},
  publisher={IEEE}
}

@inproceedings{huang2019celebrities,
  title={Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification},
  author={Huang, Yan and Wu, Qiang and Xu, Jingsong and Zhong, Yi},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2019},
  organization={IEEE}
}
