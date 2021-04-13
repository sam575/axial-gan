"""
Dataset class implementation for ARL-VTF dataset
"""
import os
from data.base_dataset import BaseDataset, get_params, get_transform, cutout
from data.image_folder import make_dataset
from PIL import Image
import pandas as pd
import numpy as np
import glob
import ipdb, pdb

def cast_columns(df: pd.DataFrame, column_dtypes):
    for column, dtype in column_dtypes:
        df[column] = df[column].astype(dtype)
    return df

def read_landmarks_csv(landmarks_fp):
    df = pd.read_csv(landmarks_fp, index_col=0)
    df = cast_columns(df, column_dtypes=[('timestamp', 'datetime64[ns]'),
                                         ('camera', 'category'),
                                         ('condition', 'category'),
                                         ('eyewear', 'category'),
                                         ('subid', 'string'),
                                         ('filepath', 'string')])
    return df

def remove_pose(A_paths, B_paths):
    A_paths = [x for x in A_paths if 'pose' not in x]
    B_paths = [x for x in B_paths if 'pose' not in x]
    print('Removed pose images')
    return A_paths, B_paths

def make_arl_meta(opt):
    print('Preparing dataset from metadata.csv ...')

    metadata = os.path.join(opt.dataroot, 'metadata.csv')
    split_file = 'splits/dev/{}{}.txt'.format(opt.phase, opt.split)
    subjects = np.loadtxt(os.path.join(opt.dataroot, split_file), dtype=str)

    df = read_landmarks_csv(metadata)
    df = df[df['subid'].isin(subjects)]
    A_paths = df[opt.A_mode == df['camera']]['filepath']
    B_paths = df[opt.B_mode == df['camera']]['filepath']

    # remove missing pairs in metadata
    A_ids = set(os.sep.join(x.split(os.sep)[:-1]) + os.sep + '_'.join(x.split('_')[2:]) for x in A_paths)
    B_ids = set(os.sep.join(x.split(os.sep)[:-1]) + os.sep + '_'.join(x.split('_')[2:]) for x in B_paths)
    AB_ids = A_ids.intersection(B_ids)
    A_paths = [os.path.join(opt.dir_A, os.sep.join(x.split(os.sep)[:-1]), opt.A_mode + '_' + x.split(os.sep)[-1]) for x in AB_ids]
    B_paths = [os.path.join(opt.dir_B, os.sep.join(x.split(os.sep)[:-1]), opt.B_mode + '_' + x.split(os.sep)[-1]) for x in AB_ids]

    assert (len(A_paths) == len(B_paths))

    # A_paths.sort()
    # B_paths.sort()

    # remove missing pairs in filesystem
    remove_ind = []
    remove_A = []
    for i, (a, b) in enumerate(zip(A_paths, B_paths)):
        if not (os.path.exists(a) and os.path.exists(b)):
            remove_ind.append(i)
            remove_A.append(a)

    print('Pairs not in filesystem:', remove_A)
    print('{} arl pairs dont exist'.format(len(remove_ind)))

    for index in sorted(remove_ind, reverse=True):
        del A_paths[index]
        del B_paths[index]

    return A_paths, B_paths

def make_arl_files(opt, dir_A, dir_B):
    print('Preparing dataset...')
    print('Recusrsing through the following directories:')
    print(dir_A, dir_B)

    A_paths = glob.glob(os.path.join(dir_A, '**/{}*.png'.format(opt.A_mode)), recursive=True)
    B_paths = glob.glob(os.path.join(dir_B, '**/{}*.png'.format(opt.B_mode)), recursive=True)

    # remove invalid pairs
    A_ids = set(os.sep.join(x.split(os.sep)[-3:-1]) + os.sep + '_'.join(os.path.basename(x).split('_')[2:]) for x in A_paths)
    B_ids = set(os.sep.join(x.split(os.sep)[-3:-1]) + os.sep + '_'.join(os.path.basename(x).split('_')[2:]) for x in B_paths)
    AB_ids = A_ids.intersection(B_ids)
    A_paths = [os.path.join(dir_A, os.sep.join(x.split(os.sep)[:-1]), opt.A_mode + '_' + x.split(os.sep)[-1]) for x in AB_ids]
    B_paths = [os.path.join(dir_B, os.sep.join(x.split(os.sep)[:-1]), opt.B_mode + '_' + x.split(os.sep)[-1]) for x in AB_ids]

    assert (len(A_paths) == len(B_paths))
    A_paths.sort()
    B_paths.sort()

    return A_paths, B_paths
    

class AlignedArlDataset(BaseDataset):
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
        parser.set_defaults(dataroot='../../dataset/Odin3')
        parser.add_argument('--dir_A', default='../../dataset/Odin3/16_bicubic', help='[8_rec, 24_rec, 16_bicubic, rec_bicubic, cropped_images]')
        parser.add_argument('--dir_B', default='../../dataset/Odin3/cropped_images', help='[rec_bicubic, cropped_images]')
        parser.add_argument('--A_mode', default='FLIR_USB', help='choose modality. [FLIR_USB, BAS_RGB2]')
        parser.add_argument('--B_mode', default='BAS_RGB2', help='choose modality. [FLIR_USB, BAS_RGB2]')
        parser.add_argument('--split', type=int, default=0, help='split id')
        parser.add_argument('--dir_A_rec', default='../../dataset/Odin3/rec_bicubic', help='[rec_bicubic] Upsampled LR image dir')
        parser.add_argument('--no_pose', default=True, action='store_true', help='No pose images while training')

        # Caution
        if parser.get_default('phase') == 'train' and parser.get_default('dir_A') != '../../dataset/Odin3/16_bicubic':
            if 'org' not in parser.get_default('dir_A'):
                parser.set_defaults(preprocess='resize_and_crop')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dir_A, 'dev' if opt.phase != 'test' else opt.phase)  # get the image directory
        self.dir_B = os.path.join(opt.dir_B, 'dev' if opt.phase != 'test' else opt.phase)  # get the image directory
        self.A_paths, self.B_paths = make_arl_meta(opt) if opt.phase != 'test' else make_arl_files(opt, self.dir_A, self.dir_B)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.dir_A_rec = os.path.join(opt.dir_A_rec, 'dev' if opt.phase != 'test' else opt.phase)
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

        assert (A_path.split(os.sep)[-1].split('_')[2:] == B_path.split(os.sep)[-1].split('_')[2:]) # filename suffix
        assert (A_path.split(os.sep)[-3] == B_path.split(os.sep)[-3]) # subject id

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # Send matlab upsampled image to discriminator
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
