"""
Training with validation script.
Monitors PSNR, SSIM, and cosine based identity score.
"""
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
from util.visualizer_tf import Visualizer
from util.metrics import *
from util.ssim import SSIM, MS_SSIM
from util import html
from util.visualizer import save_merged_images
from collections import OrderedDict
import util.util as util
from models.networks_1 import IdentityLoss
import torch.nn.functional as F
from torch import nn
import ipdb

denormalize = lambda x: (x + 1)/2.0

ssim = SSIM(window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False).cuda()
msssim = MS_SSIM(window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False).cuda()

val_psnr = 0.0
val_ssim = 0.0

def debug_vis():
    if hasattr(model, 'conf_maps'):
        for i, x in enumerate(model.conf_maps):
            x = util.tensor2im(x, denormalize=False)
            visualizer.writer.add_image('conf_{}'.format(int(128 / (2 ** i))), x, global_step=total_iters, dataformats='HWC')
            visualizer.writer.add_histogram('conf_hist_{}'.format(int(128 / (2 ** i))), x.flatten(), total_iters)

    if hasattr(model, 'out_maps'):
        for i, x in enumerate(model.out_maps[1:]):
            x = util.tensor2im(x)
            visualizer.writer.add_image('out_{}'.format(int(64 / (2 ** i))), x, global_step=total_iters, dataformats='HWC')

    if hasattr(model, 'attn_maps'):
        for i, x in enumerate(model.attn_maps):
            x = util.tensor2im(x, denormalize=False)
            visualizer.writer.add_image('attn_{}'.format(int(64 / (2 ** i))), x, global_step=total_iters, dataformats='HWC')

@torch.no_grad()
def run_val():
    global val_psnr, val_ssim, val_id
    start_time = time.time()
    if val_opt.eval:
        model.eval()

    cnt = 0
    mean_psnr = 0
    mean_ssim = 0
    mean_id = 0
    # mean_msssim = 0
    num_images = min(val_opt.num_test, len(val_dataset))
    val_iters = int(np.ceil(num_images/val_opt.batch_size))
    print('{} iterations'.format(val_iters))
    for i, data in enumerate(val_dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        n = len(img_path)
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th val iteration... %s' % (i, img_path[0]))

        # Save 1 image per iteration
        ind = 0
        visual = {k: v[ind][None] for k, v in visuals.items()}
        if visual['real_A'].shape[-1] < visual['real_B'].shape[-1]:
            # visual['real_A'] = F.interpolate(visual['real_A'], scale_factor=8)
            visual['real_A'] = data['A_rec'][ind][None]
        save_path = 'Epoch{}_{}'.format(epoch, os.path.basename(img_path[ind]))
        save_merged_images(webpage, visual, [save_path], aspect_ratio=val_opt.aspect_ratio, width=val_opt.display_winsize)
        for label, image in visual.items():
            image_numpy = util.tensor2im(image)
            visualizer.writer.add_image('val_' + label, image_numpy, global_step=(epoch*val_iters)/opt.val_freq + i, dataformats='HWC')

        fake, real = denormalize(visuals['fake_B']), denormalize(visuals['real_B'])
        mean_psnr += psnr(fake, real) * n
        mean_ssim += ssim(fake, real).mean().item() * n
        # mean_msssim += msssim(fake, real).mean().item() * n
        if val_opt.val_identity:
            mean_id += id_loss(upsample(visuals['fake_B']), upsample(visuals['real_B'])) * n

        cnt += n
        if cnt >= num_images:
            break

    mean_psnr = mean_psnr / cnt
    mean_ssim = mean_ssim / cnt
    mean_id = mean_id / cnt
    print('Num images processed:', cnt)
    s = 'Epoch: {}, Mean PSNR: {:.3f}, Mean SSIM: {:.3f}'.format(epoch, mean_psnr, mean_ssim)
    best_s = 'Prev Best Val PSNR: {:.3f}, Val SSIM: {:.3f}'.format(val_psnr, val_ssim)
    if val_opt.val_identity:
        best_s += ', Val Id: {:.3f}'.format(val_id)
    print(best_s)
    webpage.save()
    visualizer.writer.add_scalar('Val_SSIM', mean_ssim, epoch)
    visualizer.writer.add_scalar('Val_PSNR', mean_psnr, epoch)

    if val_opt.val_identity:
        s += ', Mean Id: {:.3f}'.format(mean_id)
        visualizer.writer.add_scalar('Val_ID', mean_id, epoch)
        if mean_id <= val_id:
            val_id = mean_id
            save_name = 'best_id'
            if opt.epoch_count > 1:
                save_name += '_{}'.format(opt.epoch_count)
            model.save_networks(save_name)

    if mean_psnr >= val_psnr:
        val_psnr = mean_psnr
        save_name = 'best'
        if opt.epoch_count > 1:
            save_name += '_{}'.format(opt.epoch_count)
        model.save_networks(save_name)

    if mean_ssim >= val_ssim:
        val_ssim = mean_ssim
        save_name = 'best_ssim'
        if opt.epoch_count > 1:
            save_name += '_{}'.format(opt.epoch_count)
        model.save_networks(save_name)

    print(s)
    print('Val time: {:.2f}s'.format(time.time() - start_time))
    with open(os.path.join(opt.checkpoints_dir, opt.name, 'val_log.txt'), 'a') as f:
        f.write(s + '\n')

    # Restore to training mode
    model.train()
    return

opt = TrainOptions().parse()   # get training options
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)    # get the number of images in the dataset.
print('The number of training images = %d' % dataset_size)

# Validation code
val_opt = TestOptions().parse()  # Uses only default options
val_opt.phase = 'val'
val_opt.num_test = 5000  # Run on a subset of val images
val_opt.no_flip = True
val_opt.display_id = -1
val_dataset = create_dataset(val_opt)
if val_opt.val_identity:
    id_loss = IdentityLoss(val_opt, ['maxp_5_3'], [1.0], 'cosine').cuda()
    val_id = float('inf')
    # upsample = nn.UpsamplingBilinear2d(size=(224,224))
    upsample = nn.Identity()
print('Size of validation dataset = %d' % len(val_dataset))
print('Number of val images:', val_opt.num_test)

# create a website for val
web_dir = os.path.join(opt.checkpoints_dir, opt.name, '{}_{}'.format(val_opt.phase, opt.epoch_count))
print('creating web directory', web_dir)
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch_count))

now = time.strftime("%c")
with open(os.path.join(opt.checkpoints_dir, opt.name, 'val_log.txt'), 'a') as f:
    f.write(now + '\n')

# Training model
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
total_iters = int((opt.epoch_count - 1) * (dataset_size // opt.batch_size) * opt.batch_size)

# Log GAN architecture
with open(os.path.join(opt.checkpoints_dir, opt.name, 'arch.txt'), 'w') as f:
    f.write('Generator Architecture\n')
    f.write(model.netG.module.__str__())
    f.write('\nDiscriminator Architecture\n')
    f.write(model.netD.module.__str__())
    f.close()

train_start_time = time.time()
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visual = OrderedDict([('real_A', util.tensor2im(model.real_A)),
                                  ('fake_B', util.tensor2im(model.fake_B)),
                                  ('real_B', util.tensor2im(model.real_B))])
            visualizer.display_current_results(visual, epoch, total_iters)
            debug_vis()

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time)
            visualizer.print_current_errors(epoch, epoch_iter, losses, t_comp)
            visualizer.plot_current_errors(losses, total_iters)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()

        # if i > 5:
        #     break
    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    if epoch % opt.val_freq == 0:
        print('Running Validation')
        run_val()
        print('Validation done')

print('Training was successfully finished.')
print('Total training time: {:.3f}hrs'.format((time.time() - train_start_time)/(60*60)))