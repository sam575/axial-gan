"""
Test SSIM, PSNR, and Face verification for ARL3 dataset.
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

from util.vggface import VGGFace as VggFace

from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
from PIL import Image
import pandas as pd

denormalize = lambda x: (x + 1)/2.0

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
    epoch_map = ['latest','best','best_ssim','best_id','best_psnr']
    if opt.epoch in epoch_map:
        epoch = epoch_map.index(opt.epoch)
    else:
        epoch = opt.epoch
    writer.add_scalar('Test_SSIM', mean_ssim, epoch)
    writer.add_scalar('Test_PSNR', mean_psnr, epoch)
    writer.close()

###################################################################################
# Face Verification metrics
split_dir = os.path.join(opt.dataroot, 'splits')
device = model.device
gallery_root = opt.vis_dir
probe_root = opt.dir_A

vgg = VggFace()
vgg.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, "vggface.pth")))
vgg.to(device)
vgg.eval()

data_transforms = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

pix2pix_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def display_result(fpr, tpr, thresholds, roc_auc):
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    f = interp1d(fpr, tpr)

    l1 = 'AUC: %f ' % (roc_auc)
    l2 = 'EER: %f ' % (eer)
    l3 = "FAR=0.01: %f " % (f(0.01))
    l4 = "FAR=0.05: %f " % (f(0.05))

    print(l1, l2, l3, l4, sep='\n')
    ret = [roc_auc, eer, f(0.01), f(0.05)]
    ret = [round(x * 100, 2) for x in ret]
    return ret

def get_id(x):
    return x.split('_')[2][:-1]

def remove_pose(paths):
    paths = [x for x in paths if '_p_' not in x]
    print('Removed pose images')
    return paths

def compute_probe():
    print('Computing embeddings of', probe_imgs)

    fp = open(probe_imgs).readlines()
    if opt.no_pose:
        fp = remove_pose(fp)
    count_num = len(fp)
    outputs_probe = np.zeros((count_num, 25088))
    namelist_probe = []

    for cnt, line in enumerate(tqdm(fp)):
        A_data = line.rstrip().replace('V1', opt.A_mode)
        A_path = os.path.join(probe_root, A_data)
        name = get_id(A_data)

        A_img = Image.open(A_path)
        var_A = pix2pix_transform(A_img).unsqueeze(0)
        var_A = var_A.to(device)
        d1_A = model_A(var_A)

        feat_probe = vgg(F.interpolate(d1_A, size=(224, 224), mode='bilinear', align_corners=True))[-1]
        outputs_probe[cnt, :] = feat_probe.view(1, -1).cpu().data[0]
        namelist_probe.append(name)

    return outputs_probe, namelist_probe

def compute_gallery():
    print('Computing embeddings of', gallery_imgs)

    fp = open(gallery_imgs).readlines()
    if opt.no_pose:
        fp = remove_pose(fp)
    count_num = len(fp)
    outputs_gallery = np.zeros((count_num, 25088))
    namelist_gallery = []

    for cnt, line in enumerate(tqdm(fp)):
        B_data = line.rstrip()
        B_path = os.path.join(gallery_root, B_data)
        name = get_id(B_data)

        B_img = Image.open(B_path).convert('RGB')
        var_B = data_transforms(B_img).unsqueeze(0)

        var_B = var_B.to(device)
        feat_gallery = vgg(F.interpolate(var_B, size=(224, 224), mode='bilinear', align_corners=True))[-1]
        outputs_gallery[cnt, :] = feat_gallery.view(1, -1).cpu().data[0]
        namelist_gallery.append(name)

    return outputs_gallery, namelist_gallery


def compute_result():
    similary_matrix = np.asarray(cosine_similarity(outputs_probe, outputs_gallery))
    gt_matrix = np.zeros([similary_matrix.shape[0], similary_matrix.shape[1]])
    for i in range(similary_matrix.shape[0]):
        for j in range(similary_matrix.shape[1]):
            if namelist_probe[i] == namelist_gallery[j]:
                gt_matrix[i, j] = 1
            else:
                gt_matrix[i, j] = -1

    # save result matrix
    gallery_id = os.path.splitext(gallery_imgs.split("/")[-1])[0]
    probe_id = os.path.splitext(probe_imgs.split("/")[-1])[0]

    pd.DataFrame(gt_matrix).to_csv(os.path.join(sim_matrix_folder, "gt_" + gallery_id + "_" + probe_id + ".csv"))
    pd.DataFrame(similary_matrix).to_csv(
        os.path.join(sim_matrix_folder, "pred_" + gallery_id + "_" + probe_id + ".csv"))

    print(gallery_id, probe_id)
    print("Gallery size:", len(namelist_gallery))
    print("Probe size:", len(namelist_probe))

    # compute scores
    fpr, tpr, thresholds = roc_curve(gt_matrix.reshape(-1), similary_matrix.reshape(-1))
    roc_auc = auc(fpr, tpr)
    roc_data[(gallery_id, probe_id)] = (fpr, tpr, thresholds)

    row = display_result(fpr, tpr, thresholds, roc_auc)
    if not opt.dont_save_metrics:
        with open(verf_file, 'a') as f:
            csvwriter = csv.writer(f)
            row = ['{},{}'.format(gallery_id, probe_id)] + row
            csvwriter.writerow(row)

    # plt.imshow(gt_matrix)
    # plt.show()
    # plt.imshow(similary_matrix)
    # plt.show()

mxroot = os.path.join(opt.results_dir, opt.name)
verf_file = os.path.join(mxroot, 'all_metrics.csv')

# gallery data remains same
gallery_ids = [opt.split]
gallery_data = {}
gallery_dir = os.path.join(opt.checkpoints_dir, opt.dataset_mode.split('_')[-1])
os.makedirs(gallery_dir, exist_ok=True)
pose_str = ""
if opt.no_pose:
    pose_str = "_nopose"
gallery_feats_path = os.path.join(gallery_dir, 'gallery_feats_{}{}_s{}.npy'.format(opt.A_mode, pose_str, opt.split))
if os.path.exists(gallery_feats_path):
    print('Loading gallery data')
    gallery_data = np.load(gallery_feats_path, allow_pickle=True).item()
else:
    for id_ in gallery_ids:
        gallery_imgs = os.path.join(split_dir, 'gallery_{}.txt'.format(id_))
        gallery_data[id_] = compute_gallery()
    print('Saving gallery data')
    np.save(gallery_feats_path, gallery_data)

probe_ids = [opt.split]

sim_matrix_folder = os.path.join(mxroot, 'similarity_matrix_' + opt.epoch)
os.makedirs(sim_matrix_folder, exist_ok=True)

model_A = model.netG
model_A.eval()

if not opt.dont_save_metrics:
    now = time.strftime("%c")
    with open(verf_file, 'a') as f:
        csvwriter = csv.writer(f)
        rows = [['Verification metrics', now, 'Epoch_{}'.format(opt.epoch)],
                ['Protocol', 'AUC', 'EER', 'FAR=1%', 'FAR=5%']]
        csvwriter.writerows(rows)

probe_data = {}
for id_ in probe_ids:
    probe_imgs = os.path.join(split_dir, 'probe_{}.txt'.format(id_))
    probe_data[id_] = compute_probe()

roc_data = {}
# compute result for all combinations of gallery and probe
for gid in gallery_data.keys():
    for pid in probe_data.keys():
        gallery_imgs = os.path.join(split_dir, 'gallery_{}.txt'.format(gid))
        probe_imgs = os.path.join(split_dir, 'probe_{}.txt'.format(pid))

        outputs_gallery, namelist_gallery = gallery_data[gid]
        outputs_probe, namelist_probe = probe_data[pid]
        compute_result()

roc_path = os.path.join(mxroot, 'roc_' + opt.epoch + '.npy')
np.save(roc_path, roc_data)
