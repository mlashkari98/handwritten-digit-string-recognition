# Hand-written Digit String Recognition

## Motivation
Hand-written digit string recognition (HDSR) is a computer vision task that involves identifying and recognizing a sequence of digits written by hand in an image. Unlike single-digit recognition, which classifies an image containing one digit, HDSR requires detecting and correctly recognizing multiple consecutive digits, often written without clear separation between them. It is commonly used in applications such as reading handwritten postal codes, bank cheque amounts, identification numbers, and forms.

## Architecture

Our architecture is CRNN trained using the CTC loss function which is a pytorch implementation of the model proposed in the paper ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (2016), Baoguang Shi et al."](http://arxiv.org/abs/1507.05717).

## Pretrained Model

We used an RCNN model pretrained on [Synth90k](http://www.robots.ox.ac.uk/~vgg/data/text/) dataset. The weights saved at `checkpoints/crnn_synth90k.pt`.

## Dataset

We used the ORAND-CAR-2014 dataset containing images of hand written digits sequences. The dataset is available [Here](https://www.orand.cl/icfhr2014-hdsr/) for download.

<img width="93" height="46" alt="a_car_000255" src="https://github.com/user-attachments/assets/aa56f9ce-3cf3-4c77-8b3c-d9cc3109c453" />

<img width="222" height="57" alt="a_car_002014" src="https://github.com/user-attachments/assets/1e2cffa4-0834-4f88-bbe0-72c0193c0788" />

<img width="186" height="66" alt="b_car_012484" src="https://github.com/user-attachments/assets/c3a59556-0a20-41a4-8b21-e90962583d48" />

<img width="150" height="56" alt="b_car_011748" src="https://github.com/user-attachments/assets/c91d8fe6-f399-4f25-aa63-2c232d95194d" />



## Evaluation

We chose `prefix_beam_search (beam_size=10)` as our decoding method.

| Training Method                     | Sequence Accuracy |
| ----------------------------------- | ----------------- |
| Freezing the pre-trained parameters | 0.4351            |
| Train the model from scratch        | 0.8669            |
| fine-tuning all layers              | 0.9010            |

## Train your model

You could adjust hyper-parameters in `./src/config.py`.

And train crnn models,

```command
$ python src/train.py
```

## Acknowledgement

This repo is a fork of [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch).
