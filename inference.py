import glob
import os

import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
import params
import argparse

import matplotlib.pyplot as plt


class Inference:
    def __init__(self, model_path):
        self.model_path = model_path

        nclass = len(params.alphabet) + 1
        self.model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        if params.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.converter = utils.strLabelConverter(params.alphabet)

        self.transformer = dataset.resizeNormalize((100, 32))

        print('loading pretrained model from %s' % model_path)
        self.model.load_state_dict(torch.load(model_path))

    def do_inference(self, image_path_list, plot_needed=False):
        result = []

        for image_path in image_path_list:
            image = Image.open(image_path).convert('L')
            image = self.transformer(image)
            if torch.cuda.is_available():
                image = image.cuda()
            image = image.view(1, *image.size())
            image = Variable(image)

            self.model.eval()
            preds = self.model(image)

            preds_copy = torch.tensor(preds)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            preds_size = Variable(torch.LongTensor([preds.size(0)]))
            raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
            sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)

            result.append({'path': os.path.basename(image_path),
                           'raw_pred': raw_pred,
                           'sim_pred': sim_pred})

            if plot_needed:
                self.show_prob(torch.exp(preds_copy.data), max_indices=preds.data, max_text=sim_pred)

        return result

    def show_prob(self, probs, max_indices, max_text):
        probs = probs.squeeze().transpose(1, 0)

        fig, axes = plt.subplots(len(max_indices), 1, figsize=(10, 40))
        for i, channel_index in enumerate(set(max_indices)):
            prob = probs[channel_index]
            plt.tight_layout()
            axes[i].plot(list(prob))
            axes[i].set_title('prob of channel {} ({})'.format(max_text[channel_index], channel_index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='path of an image to predict')
    parser.add_argument('--images_dir', type=str, help='directory with images to predict')
    parser.add_argument('--model_path', type=str, default=None, help='path to the model file')
    args = parser.parse_args()

    # load model
    infer = Inference(args.model_path)

    if args.images_dir:
        _result = infer.do_inference(glob.glob(os.path.join(args.images_dir, '*')))
    elif args.image_path:
        _result = infer.do_inference([args.image_path])
    else:
        raise Exception('One of images_dir or image_path should be specified')

    for item in _result:
        print(item['path'], item['raw_pred'], item['sim_pred'])
