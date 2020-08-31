# OBJECT CLASSIFIER 

## Clone this repository
- for SSH
```
git clone git@github.com:1chimaruGin/Object-classifier.git
```

- for https
```
https://github.com/1chimaruGin/Object-classifier.git
```

## Requirements
```
pip install -U requirements.txt
```

## Dataset

- the dataset directory should be the following format.

```

# for example, dog vs cat classification
data/
    -train/
        dog/
            -*.jpg or *.png
        cat/
            -*.jpg or *.png
    -val/
        dog/
            -*.jpg or *.png
        cat/
            -*.jpg or *.png
```

## Usage
```
cd Object-classifier
```
- Update number of classes and names in data.yaml

- For training model(ResNet)

```
$ python main.py -m [mode: train] -opt [optimizer: (default='SGD')]  -epochs [epochs: (default=25)] 
```
- For training model(EfficientNet)

```
$ python main.py -m [mode: train] -opt [optimizer]  -epochs [epochs] -backbone [backbone: efficientNet] -lvl [efficientNet level]
```
- For prediction
```
$ python main.py -m [mode: predict] -im [input image] - backbone [backbone: ResNet or efficientNe] -lvl [efficientNet level]
```

