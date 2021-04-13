import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from . import networks
from .networks_1 import IdentityLoss
import ipdb


class Pix2PixConfModel(BaseModel):
    """ This class implements the Axial-GAN model, for learning a mapping from input LR thermal images to
    output HR Visible images given paired data.

    By default, it uses a '--netG axial' Axial-Generator,
    a '--netD full_axial' Axial-Discriminator,
    and a '--gan_mode' hinge GAN loss.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the paper
        parser.set_defaults(norm='spectral', init_type='xavier', dataset_mode='aligned_arl')
        parser.set_defaults(preprocess='none')
        parser.set_defaults(netG='axial')  # [axial, resnet_6blocks]
        parser.set_defaults(netD='full_axial')  # [full_axial, multi, basic]
        if is_train:
            parser.set_defaults(lr=0.0002)
            parser.set_defaults(batch_size=32)
            parser.set_defaults(no_dropout=False)
            parser.set_defaults(pool_size=0, gan_mode='hinge')
            parser.add_argument('--num_D', type=int, default=1, help='Num discriminators')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_percep', type=float, default=10.0, help='weight for Perceptual loss')
            parser.add_argument('--lambda_d_feat', type=float, default=10.0, help='weight for Discriminator feature loss')
            parser.add_argument('--lambda_identity', nargs=2, type=float, default=[2, 0.2], help='weights for Identity loss')
            parser.add_argument('--L1_loss', default=True, action='store_true', help='Use L1 loss')
            parser.add_argument('--perceptual_loss', default=True, action='store_true', help='Use Perceptual loss')
            parser.add_argument('--d_feat_loss', default=True, action='store_true', help='Use Discriminator feature loss')
            parser.add_argument('--identity_loss', default=False, action='store_true', help='Use Identity loss')
            parser.add_argument('--TTUR', default=True, action='store_true', help='Update Discriminaotr faster')
            parser.add_argument('--random_rotate', type=float, default=0.0, help='Range for random rotate. Only considers POSITIVE values')
            # only valid for spectral norm and no conf guide
            parser.add_argument('--multi_scale', default=False, action='store_true', help='Multi-scale L1 loss for generator')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.num_D = opt.num_D
            if self.num_D > 1:
                assert opt.netD == 'multi'
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.num_D,
                                          opt.d_feat_loss)

        if self.isTrain:
            # define loss functions
            # self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionGAN = networks.GANLossHD(opt.gan_mode).to(self.device)
            if opt.L1_loss:
                self.criterionL1 = torch.nn.SmoothL1Loss()
                self.loss_names += ['G_L1']

            if opt.perceptual_loss:
                self.criterionVGG = networks.VGGLoss().to(self.device)
                self.loss_names += ['G_Percep']

            if opt.d_feat_loss:
                self.criterionFeat = torch.nn.L1Loss()
                self.loss_names += ['G_DFeat']

            if opt.identity_loss:
                self.criterionId = IdentityLoss(opt, ['conv_2_2', 'conv_4_2'], opt.lambda_identity).to(self.device)
                # self.criterionId = IdentityLoss(opt, ['maxp_5_3'], [2.0], 'cosine').to(self.device)
                self.loss_names += ['G_Identity']

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            lr_d = opt.lr * 2 if opt.TTUR else opt.lr
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.multi_scale_weights = [1.0, 0.5, 0.25]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if 'A_rec' in input:
            self.real_A_rec = input['A_rec'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        ret = self.netG(self.real_A)  # G(A)
        if isinstance(ret, tuple):
            # conf guide
            self.out_maps = ret[0]
            self.conf_maps = ret[1]
            self.fake_B = self.out_maps[0]
            # attn
            if len(ret) > 2:
                self.attn_maps = ret[2]
        elif isinstance(ret, list):
            # multi scale
            self.out_maps = ret
            self.fake_B = self.out_maps[0]
        else:
            self.out_maps = [ret]
            self.fake_B = ret

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Imp stuff here
        # Treat input map as of 128 size
        if self.real_A.shape[-1] < self.real_B.shape[-1]:
            self.real_A, self.real_A_rec = self.real_A_rec, self.real_A

        self.compute_maps()

        pred_fake, pred_real = self.discriminate(for_discriminator=True)

        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        pred_fake, pred_real = self.discriminate(for_discriminator=False)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True, for_discriminator=False)

        # Second, G(A) = B
        self.loss_G_L1 = 0
        if self.opt.multi_scale:
            self.loss_G_L1 = self.compute_l1_multi_scale() * self.opt.lambda_L1
        elif self.opt.L1_loss:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # Perceptual loss
        if self.opt.perceptual_loss:
            self.loss_G_Percep = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_percep
            self.loss_G += self.loss_G_Percep

        # Discriminator feature loss
        if self.opt.d_feat_loss:
            # ipdb.set_trace()
            num_D = len(pred_fake)
            self.loss_G_DFeat = 0
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    self.loss_G_DFeat += unweighted_loss * self.opt.lambda_d_feat / num_D
            self.loss_G += self.loss_G_DFeat

        # Identity loss
        if self.opt.identity_loss:
            self.loss_G_Identity = self.criterionId(self.fake_B, self.real_B)
            self.loss_G += self.loss_G_Identity

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_maps(self):
        # Compute multi-scale maps for multi-scale generator or discriminator
        # TODO: check bilinear mode vs nearest
        mode = 'nearest'
        self.real_B_maps = [self.real_B]
        for i in range(1, max(len(self.out_maps), self.num_D)):
            self.real_B_maps.append(F.interpolate(self.real_B, scale_factor=1 / (2 ** i), mode=mode))

        self.real_A_maps = [self.real_A]
        for i in range(1, self.num_D):
            self.real_A_maps.append(F.interpolate(self.real_A, scale_factor=1 / (2 ** i), mode=mode))

            if i >= len(self.out_maps):
                self.out_maps.append(F.interpolate(self.fake_B, scale_factor=1 / (2 ** i), mode=mode))

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.
    def discriminate(self, for_discriminator=True):
        fake_and_real = []
        for i in range(self.num_D):
            input_semantics, fake_image, real_image = self.real_A_maps[i], self.out_maps[i], self.real_B_maps[i]
            if for_discriminator:
                fake_image = fake_image.detach()

            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)

            # In Batch Normalization, the fake and real images are
            # recommended to be in the same batch to avoid disparate
            # statistics in fake and real images.
            # So both fake and real images are fed to D all at once.
            # fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
            fake_and_real.append(torch.cat([fake_concat, real_concat], dim=0))

        # For single or other discrimnators
        if self.num_D == 1:
            fake_and_real = fake_and_real[0]

        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def compute_l1_multi_scale(self):
        weights = self.multi_scale_weights
        loss = 0
        for x, t, w in zip(self.out_maps, self.real_B_maps, weights):
            loss += self.criterionL1(x, t) * w
        return loss
