"""Usage: predict.py [-m MODEL] [-s BS] [-d DECODE] [-b BEAM] [IMAGE ...]

-h, --help    show this
-m MODEL     model file [default: ./checkpoints/crnn_synth90k.pt]
-s BS       batch size [default: 256]
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]
-b BEAM   beam size [default: 10]

"""
from docopt import docopt
import torch
from tqdm import tqdm

from config import common_config
from model import CRNN
from ctc_decoder import ctc_decode


def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images = data.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')

