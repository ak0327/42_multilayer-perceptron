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
usage: dataloader.py [-h] --dataset_path DATASET_PATH [--train_size TRAIN_SIZE] [--shuffle SHUFFLE] --save_npz SAVE_NPZ --save_dir SAVE_DIR

Split WDBC dataset into train and test sets

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to the WBDC CSV dataset
  --train_size TRAIN_SIZE
                        Percentage of training division (float in (0.0, 1.0))
  --shuffle SHUFFLE     Whether to shuffle the data before splitting (true/false, t/f)
  --save_npz SAVE_NPZ   Save to train.npz and test.npz, othewise csv (true/false, t/f)
  --save_dir SAVE_DIR   dataset save dir
```

#### Run
```shell
$ docker compose exec ft_mlp \
  python3 srcs/dataloader.py \
	--dataset_path data/data.csv \
	--train_size 0.8 \
	--shuffle true \
	--save_npz false \
	--save_dir data
```

### iii) Train
#### Usage
```shell
usage: train.py [-h] --dataset_path DATASET_PATH [--hidden_features HIDDEN_FEATURES] [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                [--weight_decay WEIGHT_DECAY] [--optimizer OPTIMIZER] [--verbose VERBOSE] [--plot PLOT] [--metrics_interval METRICS_INTERVAL]
                [--patience PATIENCE] --save_dir SAVE_DIR

Train on WDBC dataset with MLP

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to the train WBDC CSV or NPZ dataset.
  --hidden_features HIDDEN_FEATURES
                        Number of neurons in each hidden layer (2 to 5 values, e.g., --hidden_features 24 42 or --hidden_features 24 42 24 42 24)
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
$ docker compose exec ft_mlp \
  python3 srcs/train.py \
	--dataset_path data/data_train.csv \
	--hidden_features "50 30" \
	--epochs 1000 \
	--learning_rate 0.0001 \
	--optimizer Adam \
	--verbose true \
	--plot true \
	--metrics_interval 100 \
	--save_dir data
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
                        Path to the predict WBDC CSV or NPZ dataset.
```

#### Run
```shell
$ docker compose exec ft_mlp \
  python3 srcs/predict.py \
	--model_path data/model.pkl \
	--dataset_path data/data_test.csv
```


## 2-3) Detailed Information
* For more details, please refer to [./notebook/mlp.ipynb](./notebook/review.ipynb)
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
