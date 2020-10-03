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

## Usage with Argparse
```
cd Object-classifier/objifier
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

## Usage with YAML (via pip)

#### Create a YAML file as sample below:

- For training [train.yaml]
```
nc: 10
# names: ['mantled_howler', 'patas_monkey', 'bald_uakari', 'japanese_macaque', 'pygmy_marmoset', 
#       'white_headed_capuchin', 'silvery_marmoset', 'common_squirrel_monkey', 'black_headed_night_monkey','nilgiri_langur' ]

names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

backbone: 'efficientNet'

efftlvl: 0

optimizer: 'Adam'

mode: 'train'

epoch: 2

load: False

output: 'output'

dataset_path: null

```
- For prediction [predict.yaml]

```
nc: 10
# names: ['mantled_howler', 'patas_monkey', 'bald_uakari', 'japanese_macaque', 'pygmy_marmoset', 
#       'white_headed_capuchin', 'silvery_marmoset', 'common_squirrel_monkey', 'black_headed_night_monkey','nilgiri_langur' ]

names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

backbone: 'efficientNet'

efftlvl: 0

output: 'output'

image: 'baobao.jpg'

```

