import argparse
import torch
import matplotlib.pyplot as plt
from data import NoisyBSDSDataset
from argument import parse
from model import DnCNN
import nntools as nt
from utils import DenoisingStatsManager, plot
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import os



def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

def myim(image):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    return image

def image_preprocess(img_path):
    img = Image.open(img_path).convert('RGB')  
    transform = tv.transforms.Compose([
        tv.transforms.Resize(300),
        # convert it to a tensor
        tv.transforms.ToTensor(),
        # normalize it to the range [−1, 1]
        tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    img = transform(img)
    return img


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset
    train_set = NoisyBSDSDataset(
        args.root_dir, image_size=args.image_size, sigma=args.sigma)
    test_set = NoisyBSDSDataset(
        args.root_dir, mode='test', image_size=args.test_image_size, sigma=args.sigma)

    # model
    net = DnCNN(args.D, C=args.C).to(device)

    # optimizer
    adam = torch.optim.Adam(net.parameters(), lr=args.lr)

    # stats manager
    stats_manager = DenoisingStatsManager()

    # experiment
    exp = nt.Experiment(net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
                        output_dir=args.output_dir, perform_validation_during_training=True)

    # run
    if args.plot:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 7))
        exp.run(num_epochs=args.num_epochs, plot=lambda exp: plot(exp, fig=fig, axes=axes,
                                                noisy=test_set[73][0]))
    else:
        exp.run(num_epochs=args.num_epochs)

    model = exp.net.to(device)
    titles = ['original', 'denoised', 'noise (enhanced)']
    img_path = "test.png"
    x = image_preprocess(img_path=img_path)
    img = []
    img.append(x)
    x1=x
    x = x.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        y = model.forward(x)
    img.append(y[0])
    #img.append(x1-y[0])
    fig, axes = plt.subplots(ncols=len(img)+1, figsize=(9,5), sharex='all', sharey='all')
    myimshow(img[0], ax=axes[0])
    x0 = myim(img[0])
    axes[0].set_title(f'{titles[0]}')
    myimshow(img[1], ax=axes[1])
    x1 = myim(img[1])
    axes[1].set_title(f'{titles[1]}')
    ax = axes[2]
    h = ax.imshow((x0-x1)*5)
    ax.axis('off')
    ax.set_title(f'{titles[2]}')
    plt.savefig('img_result.png')
    

if __name__ == '__main__':
    args = parse()
    print(args)
    run(args)
