# MakeGoodFood

## Introduction

The project is about generating a fake food image and transfers it into an animation art style which would make the food looks more delicious, then give a positive food review in text.
It could be divided into three parts:
 - DCGAN
     - Generate food image.
 - fast-style-transfer
     - Transfer the food image into animation art style.
 - GPT-2 text generation
     - Generate review for the food.
---

## Preparation

 - [Python 3.7.5](https://www.python.org/downloads/release/python-375/)
     - Install Python from official website
 - [PyTorch](https://pytorch.org/get-started/locally/)
     - Install PyTorch by following instructions on official website
 - [Cuda 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
     - Install Cuda by following instructions on official website
 - Install Tensorflow 2.0.0
    ```
    $ pip install tensorflow-gpu
    ```
 - Install GPT-2 library
    ```
    $ pip install gpt-2-simple
    ```
 - Install GPT-2 335M model
    ```
    $ python gpt2_install.py
    ```
 - Download trained GPT-2 model
     - Download [here](https://drive.google.com/file/d/1-CIlewaAqMTwT01UEJnjkTgyjpZC7OKd/view?usp=sharing)
     - place under the project folder and extract it by **Extract here**, which should create a folder named **checkpoint**, and a subfolder named **pure_review** inside it.
 - Install [TensorCV](https://github.com/conan7882/DeepVision-tensorflow)
     - Install TensorCV by following the instructions
 - Install imageio
    ```
    $ pip install imageio
     ```
 - Install scipy
     ```
    pip install scipy
    ```
 - Dataset for DCGAN generating food image
     - the data we use is from [here](https://github.com/karansikka1/iFood_2019)


---

## Train a DCGAN model to generate food images

I want to train a GAN model to generate images of a particular food. But the dataset I found for each food is not large enough, so I didn't use its classes. The dataset can be found [here]( https://github.com/karansikka1/iFood_2019).
I chose DCGAN rather than GAN because I tried GAN first, but the results weren't good. DCGAN has better performance on images.
### Some results
![](https://i.imgur.com/O6AHwBj.png)  ![](https://i.imgur.com/gWPR1r6.png)

### Model
Generator has 5 layers. The original input is 100x1x1. Outputs of each layer is (ngf*8)x4x4, (ngf*4)x8x8, (ngf*2)x16x16, (ngf)x32x32 and 3x96x96 (ngf=64). The kernel size, stride and padding of each layer are (4，1，0), (4，2，1), (4，2，1), (4，2，1) and (5，3，1). The structure of discriminator is reversed. Learning rate is 0.0002, batch size is 100 and epoch is 500.

### Run
```
$ python food_generator.py --data_path your/image/data/folder/path/
```
---

## Style Transfer
Mainly reimplement from [conan7882/fast-style-transfer](https://github.com/conan7882/fast-style-transfer) and retrain our own model.
We use [this dataset](https://www.kaggle.com/vermaavi/food11) to train, while using a [food-related image](https://i.imgur.com/3NfhbhW.jpg) from [Food Wars!: Shokugeki no Soma](https://en.wikipedia.org/wiki/Food_Wars!:_Shokugeki_no_Soma)([食戟のソーマ](https://ja.wikipedia.org/wiki/%E9%A3%9F%E6%88%9F%E3%81%AE%E3%82%BD%E3%83%BC%E3%83%9E))
Here are some examples from transferring real food photo:
![](https://i.imgur.com/I0Tvi0W.png) ![](https://i.imgur.com/JiEC3ll.png)

![](https://i.imgur.com/WYI0IPK.png) ![](https://i.imgur.com/dq4uWYZ.png)

![](https://i.imgur.com/NmzFb4w.png) ![](https://i.imgur.com/JXcscvy.png)

and examples from transferring generated food photo

![](https://i.imgur.com/O6AHwBj.png) ![](https://i.imgur.com/vo7OgWO.png)

![](https://i.imgur.com/gWPR1r6.png) ![](https://i.imgur.com/itoUG6m.png)

### Run
```
$ python style_transfer.py --generate_image --input_path input/image/path --save_path output/image/path
```
---

## Generate Review

We are using GPT-2 model for generating food reviews, using [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) to train the model, while only using reviews giving 4 or 5 stars.
The program will ask you to input the prefix, which will initial the generated review.
Examples:

```
input prefix: The coffee
> The coffee was decaf, not decaf on the package.  I'd have never known it was decaf. This is not a decaf that you can buy at Starbucks

input prefix: This beef stew
> This beef stew is absolutely delicious and will go down a treat for all those who enjoy stew with the added benefit of having a good quality food source.  I have been adding some beef flavor to my own products for over 20 years and have always enjoyed the flavor.

input prefix: If this burger
> If this burger isn't as good as the original, it is far better than the original and it is even more of a hit.
```

### Run
```
$ python review_generator.py
```
