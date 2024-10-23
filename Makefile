.PHONY: all
all: build

.PHONY: build
build:
	docker compose up --build -d

.PHONY: notebook
notebook:
	docker compose exec ft_mlp jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''

.PHONY: down
down:
	docker compose down

.PHONY: info
info:
	@docker compose exec ft_mlp python --version

.PHONY: exec
exec:
	docker compose exec ft_mlp sh

.PHONY: run
run:
	docker compose exec ft_mlp python3 srcs/main.py

.PHONY: dataloader
dataloader:
	docker compose exec ft_mlp \
	python3 srcs/dataloader.py \
	--dataset_path data/data.csv \
	--test_size 0.2 \
	--shuffle true \
	--save_dir data

.PHONY: train
train:
	docker compose exec ft_mlp \
	python3 srcs/train.py \
	--dataset_path data/data_train.csv \
	--hidden_features "10 2" \
	--epochs 4000 \
	--learning_rate 1e-4 \
	--optimizer Adam \
	--verbose true \
	--plot false \
	--metrics_interval 100 \
	--save_dir data

.PHONY: predict
predict:
	docker compose exec ft_mlp \
	python3 srcs/predict.py \
	--model_path data/model.pkl \
	--dataset_path data/data_test.csv

.PHONY: test
test:
	docker compose exec ft_mlp pytest -v -c config/pytest.ini

.PHONY: clean
clean:
	docker compose down --rmi all

.PHONY: fclean
fclean:
	docker system prune -a
