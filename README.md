# multilayer-perceptron

<hr>

# 1. Overview
Re-implementation of a multilayer perceptron

<hr>

# 2. How to use
## 2-1) Set up execution environment with Docker Compose
```shell
$ make

$ make info
Python 3.10.15
```


## 2-2) Run script
### i) Run all
Run dataloader, train and predict program at once
```shell
$ make run
```

### ii) Dataloader
#### Usage
```shell
usage: dataloader.py [-h] --dataset_path DATASET_PATH [--test_size TEST_SIZE] [--shuffle SHUFFLE] --save_dir SAVE_DIR

Split WDBC dataset into train and test sets

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to the WBDC CSV dataset
  --test_size TEST_SIZE
                        Percentage of training division (float in (0.0, 1.0))
  --shuffle SHUFFLE     Whether to shuffle the data before splitting (true/false, t/f)
  --save_dir SAVE_DIR   dataset save dir
```

#### Run
```shell
$ make dataloader

or

$ docker compose exec ft_mlp \
  python3 srcs/dataloader.py \
	--dataset_path data/data.csv \
	--test_size 0.2 \
	--shuffle true \
	--save_dir data

# [Dataloader]
#  Dataset: data/data.csv
#   Splitting...
#
# train set:
#   B: 250 (62.7%)
#   M: 149 (37.3%)
# 
# test set:
#   B: 107 (62.9%)
#   M: 63 (37.1%)
#   Split dataset SUCCESS
# 
#  Data_train data saved to /app/data/data_train.csv
#  Data_test data saved to /app/data/data_test.csv
```

### iii) Train
#### Usage
```shell
usage: train.py [-h] --dataset_path DATASET_PATH [--hidden_features HIDDEN_FEATURES] [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                [--weight_decay WEIGHT_DECAY] [--optimizer OPTIMIZER] [--verbose VERBOSE] [--plot PLOT]
                [--metrics_interval METRICS_INTERVAL] [--patience PATIENCE] --save_dir SAVE_DIR

Train on WDBC dataset with MLP

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to the train WBDC CSV or NPZ dataset.
  --hidden_features HIDDEN_FEATURES
                        Number of neurons in each hidden layer in range [2, 500](2 to 5 values, e.g., --hidden_features 24 42 or
                        --hidden_features 24 42 24 42 24)
  --epochs EPOCHS       Number of training epochs (integer in range [1, 100000], default: 5000)
  --learning_rate LEARNING_RATE
                        Learning rate for training (float in range [0.0001, 1.0], default: 0.01)
  --weight_decay WEIGHT_DECAY
                        Weight decay (float in range [0.0, 1.0], default: 0.0)
  --optimizer OPTIMIZER
                        Optimizer ([SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam], default: SGD)
  --verbose VERBOSE     verbose (true/false, t/f)
  --plot PLOT           plot (true/false, t/f)
  --metrics_interval METRICS_INTERVAL
                        metrics interval in range[1, 1000]
  --patience PATIENCE   Ealry stopping patience in range[1, 10000]
  --save_dir SAVE_DIR   model save dir
```

#### Run
```shell
$ make train

or

$ docker compose exec ft_mlp \
  python3 srcs/train.py \
	--dataset_path data/data_train.csv \
	--hidden_features "29 23" \
	--epochs 2000 \
	--learning_rate 1e-4 \
	--optimizer Adam \
	--verbose true \
	--plot true \
	--metrics_interval 100 \
	--patience 250 \
	--save_dir data

# [Train]
#  Dataset: data/data_train.csv
# 
#  MODEL:
#    Sequential(
#      (0): Linear(in_features=30, out_features=29)
#      (1): ReLU()
#      (2): Linear(in_features=29, out_features=23)
#      (3): ReLU()
#      (4): Linear(in_features=23, out_features=2)
#      (5): Softmax()
#    )
#  OPTIMIZER: Adam(lr=0.0001, beta1=0.9, beta2=0.999)
#  CRITERIA : CrossEntropyLoss()
# 
#  Training WDBC...
#   X_train shape: (280, 30)
#   X_valid shape: (119, 30)
# 
#   Epoch:   0/2000, Train [Loss:0.6926, Acc:0.7500], Valid [Loss:0.6926, Acc:0.8487]
#   Epoch: 100/2000, Train [Loss:0.6142, Acc:0.9571], Valid [Loss:0.6232, Acc:0.8908]
#   Epoch: 200/2000, Train [Loss:0.4226, Acc:0.9571], Valid [Loss:0.4561, Acc:0.8908]
#   Epoch: 300/2000, Train [Loss:0.2618, Acc:0.9607], Valid [Loss:0.3108, Acc:0.9076]
#   Epoch: 400/2000, Train [Loss:0.1740, Acc:0.9750], Valid [Loss:0.2276, Acc:0.9328]
#   Epoch: 500/2000, Train [Loss:0.1240, Acc:0.9786], Valid [Loss:0.1806, Acc:0.9580]
#   Epoch: 600/2000, Train [Loss:0.0940, Acc:0.9821], Valid [Loss:0.1567, Acc:0.9580]
#   Epoch: 700/2000, Train [Loss:0.0743, Acc:0.9893], Valid [Loss:0.1430, Acc:0.9580]
#   Epoch: 800/2000, Train [Loss:0.0599, Acc:0.9929], Valid [Loss:0.1347, Acc:0.9748]
#   Epoch: 900/2000, Train [Loss:0.0490, Acc:0.9929], Valid [Loss:0.1309, Acc:0.9664]
#   Epoch:1000/2000, Train [Loss:0.0407, Acc:0.9929], Valid [Loss:0.1300, Acc:0.9664]
#   Epoch:1100/2000, Train [Loss:0.0343, Acc:0.9964], Valid [Loss:0.1307, Acc:0.9664]
#   Epoch:1200/2000, Train [Loss:0.0291, Acc:0.9964], Valid [Loss:0.1333, Acc:0.9580]
#   Early Stopping
# 
#  Training curve graph save to data/training_curve.png
# 
#  Metrics: 
#   Train - loss:0.0268 [Accuracy:0.9929, Precision:1.0000, Recall:0.9810, F1:0.9904]
#   Valid - loss:0.1352 [Accuracy:0.9664, Precision:0.9762, Recall:0.9318, F1:0.9535]
# 
#  Model data saved to /app/data/model.pkl
#  Metrics saved to /app/data/metrics.npz
```

### iv) Predict
#### Usage
```shell
usage: predict.py [-h] --model_path MODEL_PATH --dataset_path DATASET_PATH

Predict with trained model

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to trained model
  --dataset_path DATASET_PATH
                        Path to the predict WBDC CSV dataset.
```

#### Run
```shell
$ make predict

or

$ docker compose exec ft_mlp \
  python3 srcs/predict.py \
	--model_path data/model.pkl \
	--dataset_path data/data_test.csv
	
# [Predict]
#  Dataset: data/data_test.csv
# 
#  Predicting WDBC...
#   Result:
#    - Loss: 0.1179
#    - Accuracy:0.9588, Precision:0.9375, Recall:0.9524, F1:0.9449
```


## 2-3) Detailed Information
* For more details, please refer to [./notebook/review.ipynb](./notebook/review.ipynb)
* To view in Jupyter Notebook:
  * Run `make notebook`
  * Access `localhost:30000` in your browser on the host machine

<hr>

# 3. Confirmed Environments
* Ubuntu 22.04.2 LTS (ARM64)
* MacOS OS Ventura 13.5 (ARM64)

<hr>

# 4. References
* 斎藤康毅, ゼロから作るDeep Learning, オライリージャパン
* 岡谷貴之, 深層学習 改訂第2版, 講談社
* Ian Goodfellow 他, 深層学習, KADOKAWA
* [Neural Network Console Deep Learning精度向上テクニック：様々な最適化手法 #1](https://www.youtube.com/watch?v=q933reMpvX8&t=358s)
* [Gradient-Sensitive Optimization for Convolutional Neural Networks](https://onlinelibrary.wiley.com/doi/10.1155/2021/6671830)
* [【Matplotlib】グラフをリアルタイム表示する (pause, remove)](https://www.useful-python.com/matplotlib-realtime/)
