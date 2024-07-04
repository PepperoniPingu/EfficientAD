.PHONY: inference train dataset huggingface-login tensorboard

inference:
	python main.py

train:
	python train.py

dataset:
	cd dataset && $(MAKE)

huggingface-login:
	huggingface-cli login

tensorboard:
	tensorboard --logdir=runs