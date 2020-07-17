# OBJECT CLASSIFIER USING RESNET 50

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

## Usage
```
cd Object_classifier
```
- For training model

```
$ python main.py -m [mode: train] -opt [optimizer]  -epochs [epochs] -arch [backbone: ResNet or efficientNet] -lvl [efficientNet level]
```
- For prediction
```
$ python main.py -m [mode: predict] -im [input image] - arch [backbone: ResNet or efficientNe] -lvl [efficientNet level]
```

