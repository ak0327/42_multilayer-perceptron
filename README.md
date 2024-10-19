# multilayer-perceptron
## 1. Overview
Re-implementation of a multilayer perceptron

<br>

## 2. How to use
### 2-1) Install vertual environment
```shell
$ make

$ make info
Python 3.10.15
```

### 2-2) Run script
```shell
# run all
$ make run

# run dataloader
$ docker compose exec ft_mlp \
	python3 srcs/dataloader.py \
	--dataset_path data/data.csv \
	--train_size 0.8 \
	--shuffle true \
	--save_npz false \
	--save_dir data

# run train
$ docker compose exec ft_mlp \
	python3 srcs/train.py \
	--dataset_path data/data_train.csv \
	--hidden_features 50 30 \
	--epochs 1000 \
	--learning_rate 0.0001 \
	--optimizer Adam \
	--verbose true \
	--plot true \
	--metrics_interval 100 \
	--save_dir data
	
# run predict
$ docker compose exec ft_mlp \
	python3 srcs/predict.py \
	--model_path data/model.pkl \
	--dataset_path data/data_test.csv
```

### 2-3) More detail
See `notebook/mlp.ipynb`


<br>

## 3. Confirmed Environments
* Ubuntu 22.04.2 LTS (ARM64)
* MacOS OS Ventura 13.5 (ARM64)

<br>

## 4. References
* 斎藤康毅, ゼロから作るDeep Learning, オライリージャパン
* 岡谷貴之, 深層学習 改訂第2版, 講談社
* Ian Goodfellow 他, 深層学習, KADOKAWA
* [Neural Network Console Deep Learning精度向上テクニック：様々な最適化手法 #1](https://www.youtube.com/watch?v=q933reMpvX8&t=358s)
* [Gradient-Sensitive Optimization for Convolutional Neural Networks](https://onlinelibrary.wiley.com/doi/10.1155/2021/6671830)
* [【Matplotlib】グラフをリアルタイム表示する (pause, remove)](https://www.useful-python.com/matplotlib-realtime/)
