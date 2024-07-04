.PHONY: inference train dataset huggingface-login

inference:
	python main.py

train:
	python train.py

dataset:
	cd dataset && $(MAKE)

huggingface-login:
	huggingface-cli login