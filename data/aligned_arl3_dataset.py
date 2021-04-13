"""
Dataset class implementation for extended ARL polarimetric thermal dataset
"""
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import pandas as pd
import numpy as np
import glob
import ipdb, pdb

def get_id(x):
    return x.split('_')[2][:-1]

def remove_pose(A_paths, B_paths):
    A_paths = [x for x in A_paths if '_p_' not in x]
    B_paths = [x for x in B_paths if '_p_' not in x]
    print('Removed pose images')
    return A_paths, B_paths

def make_files(opt, dir_A, dir_B):
    print('Preparing dataset...')
    print('Recusrsing through the following directories:')
    print(dir_A, dir_B)

    split_file = os.path.join(opt.dataroot, 'splits', '{}_{}.txt'.format(opt.phase, opt.split))
    ids = open(split_file, 'r').read().split('\n')[:-1]


    A_paths = glob.glob(os.path.join(dir_A, '*.[pj][np]g'), recursive=True)
    B_paths = glob.glob(os.path.join(dir_B, '*.[pj][np]g'), recursive=True)
    A_ext = os.path.splitext(A_paths[0])[-1]
    B_ext = os.path.splitext(B_paths[0])[-1]

    A_ids = [os.path.splitext(os.path.basename(x))[0] for x in A_paths]
    B_ids = [os.path.splitext(os.path.basename(x))[0] for x in B_paths]
    # retain only train/test files
    A_ids = set(x.replace(opt.A_mode, 'XX') for x in A_ids if get_id(x) in ids)
    B_ids = set(x.replace(opt.B_mode, 'XX') for x in B_ids if get_id(x) in ids)
    # remove invalid pairs
    AB_ids = A_ids.intersection(B_ids)
    if opt.phase == 'test':
        AB_ids = list(AB_ids)
        AB_ids.sort()

    A_paths = [os.path.join(dir_A, x.replace('XX', opt.A_mode) + A_ext) for x in AB_ids]
    B_paths = [os.path.join(dir_B, x.replace('XX', opt.B_mode) + B_ext) for x in AB_ids]

    assert (len(A_paths) == len(B_paths))

    print("Num files in {}:{}".format(dir_A, len(A_paths)))
    # ipdb.set_trace()

    return A_paths, B_paths


class AlignedArl3Dataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataroot='../../dataset/odin_data')
        parser.add_argument('--dir_A', default='../../dataset/odin_data/Polar_16', help='[Polar_16, Polar_128_rec, S0_128_rec, S0_16]')
        parser.add_argument('--dir_B', default='../../dataset/odin_data/Vis_128', help='[Visible]')
        parser.add_argument('--A_mode', default='Polar', help='choose modality. [V1, S0, Polar]')
        parser.add_argument('--B_mode', default='V1', help='choose modality. [V1, S0, Polar]')
        parser.add_argument('--split', type=int, default=5, help='split id')
        parser.add_argument('--no_pose', default=True, action='store_true', help='No pose images while training')
        parser.add_argument('--vis_dir', default='../../dataset/odin_data/Vis_128', help='[Visible] Gallery dir')

        # Caution
        if parser.get_default('phase') == 'train':
            if parser.get_default('no_pose'):
                parser.set_defaults(n_epochs=40, n_epochs_decay=30)
            else:
                parser.set_defaults(n_epochs=50, n_epochs_decay=25)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.normpath(opt.dir_A)  # get the image directory
        self.dir_B = os.path.normpath(opt.dir_B)  # get the image directory
        self.A_paths, self.B_paths = make_files(opt, self.dir_A, self.dir_B)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # self.dir_A_rec = os.path.normpath(opt.dir_A_rec)
        # Caution:
        if opt.phase == 'train' and opt.preprocess == 'none' and '16' not in opt.dir_A:
            opt.preprocess = 'resize_and_crop'
        opt.dir_A_rec = os.path.join(opt.dataroot, '{}_128_rec'.format(opt.A_mode))
        self.dir_A_rec = opt.dir_A_rec
        if opt.no_pose:
            self.A_paths, self.B_paths = remove_pose(self.A_paths, self.B_paths)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # Send matlab reconstructed image to discriminator
        if A.shape[-1] < B.shape[-1]:
            A_rec_path = A_path.replace(self.dir_A, self.dir_A_rec)
            A_rec = Image.open(A_rec_path).convert('RGB')
            A_rec_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            A_rec = A_rec_transform(A_rec)
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_rec': A_rec}

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)