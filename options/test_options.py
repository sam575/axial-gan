from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', default=True, action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=200, help='how many test images to run')
        # rewrite devalue values
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.set_defaults(epoch='best')
        parser.add_argument('--val_identity', default=True, action='store_true', help='Use Identity loss for monitoring')
        parser.add_argument('--dont_save_metrics', default=False, action='store_true', help='Dont Save verfication and psnr metrics')
        self.isTrain = False
        return parser
