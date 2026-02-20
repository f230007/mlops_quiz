
generate:
	python src/generate_data.py

preprocess:
	python src/preprocess.py

train:
	python src/train.py

evaluate:
	python src/evaluate.py

all: generate preprocess train evaluate