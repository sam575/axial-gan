"""
Test SSIM, PSNR for a given model.
Save only the given number of synthesized pairs.
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_merged_images
from util import html
from util.metrics import *
from util.ssim import SSIM, MS_SSIM
import numpy as np
import time
import pdb, ipdb
import csv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random

denormalize = lambda x: (x + 1)/2.0

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    # opt.num_threads = 0   # test code only supports num_threads = 0
    # opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    if opt.num_test != 0:
        random.seed(0)
        random_inds = random.sample(range(len(dataset)),opt.num_test)
        random_inds.sort()

    FOLDER = os.path.join(opt.results_dir, opt.name)
    ssim = SSIM(window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False).cuda()
    msssim = MS_SSIM(window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False).cuda()

    cnt = 0
    save_cnt = 0
    mean_psnr = 0
    mean_ssim = 0
    mean_msssim = 0

    print('Length of dataset:', len(dataset))
    print(np.ceil(len(dataset)/opt.batch_size), 'iterations')
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        n = len(img_path)
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th iteration... %s' % (i, img_path[0]))

        if not opt.dont_save_metrics:
            if opt.num_test == 0:
                for j in range(n):
                    visual = {k: v[j][None] for k, v in visuals.items()}
                    if visual['real_A'].shape[-1] < visual['real_B'].shape[-1]:
                        # visual['real_A'] = F.interpolate(visual['real_A'], scale_factor=8)
                        visual['real_A'] = data['A_rec'][j][None]
                    save_merged_images(webpage, visual, [img_path[j]], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            else:
                for j in range(n):
                    if i*opt.batch_size + j in random_inds:
                        visual = {k: v[j][None] for k, v in visuals.items()}
                        if visual['real_A'].shape[-1] < visual['real_B'].shape[-1]:
                            # visual['real_A'] = F.interpolate(visual['real_A'], scale_factor=8)
                            visual['real_A'] = data['A_rec'][j][None]
                        save_merged_images(webpage, visual, [img_path[j]], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
                        save_cnt += 1

        fake, real = denormalize(visuals['fake_B']), denormalize(visuals['real_B'])
        mean_psnr += psnr(fake, real) * n
        mean_ssim += ssim(fake, real).mean().item() * n
        # mean_msssim += msssim(fake, real).mean().item() * n
        cnt += n

    mean_psnr = mean_psnr / cnt
    mean_ssim = mean_ssim / cnt
    print('Num images processed:', cnt)
    print('Num Saved images:', save_cnt)
    print('Saving at:', FOLDER)
    print('Mean PSNR: {:.3f}'.format(mean_psnr))
    print('Mean SSIM: {:.3f}'.format(mean_ssim))
    # print('Mean MS_SSIM: ', mean_msssim / cnt)

    if not opt.dont_save_metrics:
        now = time.strftime("%c")
        with open(os.path.join(FOLDER, 'all_metrics.csv'), 'a') as f:
            csvwriter = csv.writer(f)
            # rows = [['Generation metrics', now, 'Epoch_{}'.format(opt.epoch), 'No Pose: {}'.format(opt.no_pose), opt.name],
            rows = [['Generation metrics', now, 'Epoch_{}'.format(opt.epoch), opt.name],
                    ['PSNR', 'SSIM']]
            csvwriter.writerows(rows)

            row = [mean_psnr, mean_ssim]
            # row.append(mean_msssim/cnt)
            row = [round(x, 3) for x in row]
            csvwriter.writerow(row)

        webpage.save()  # save the HTML

        # Tensorboard writer
        log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
        writer = SummaryWriter(log_dir)
        epoch_map = ['latest', 'best', 'best_ssim', 'best_id', 'best_psnr']
        if opt.epoch in epoch_map:
            epoch = epoch_map.index(opt.epoch)
        else:
            epoch = opt.epoch
        writer.add_scalar('Test_SSIM', mean_ssim, epoch)
        writer.add_scalar('Test_PSNR', mean_psnr, epoch)
        writer.close()
