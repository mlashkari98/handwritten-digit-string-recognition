# Hand Written Digit string Recognition


## Architecture

Our architecture is CRNN trained using the CTC loss function which is a pytorch implementation of the model proposed in the paper ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (2016), Baoguang Shi et al."](http://arxiv.org/abs/1507.05717).

## Pretrained Model

We used an RCNN model pretrained on [Synth90k](http://www.robots.ox.ac.uk/~vgg/data/text/) dataset. The weights saved at `checkpoints/crnn_synth90k.pt`.

## Dataset

We used the ORAND-CAR-2014 dataset containing images of hand written digits sequences. The dataset is available [Here](https://www.orand.cl/icfhr2014-hdsr/) for download.


## Evaluatation

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
