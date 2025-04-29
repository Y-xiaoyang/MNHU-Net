from utils.utils import *

def parse_args():
    """Testing Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MNHU-net for IRSTD')

    # choose model
    parser.add_argument('--model', type=str, default='MNHU',
                        help='MNHU, MNHU_E, MNHU_D')
    # Deep supervision for MNHU-nets              
    parser.add_argument('--deep_supervision', type=str, default='DSV', help='DSV or None')               
    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='dataset name:  NUAA-SIRST, NUDT-SIRST, IRSTD-1k')
    parser.add_argument('--st_model', type=str, default='MNHU',                             
                        help='')                         
    parser.add_argument('--model_dir', type=str,                                                         
                        default = 'MNHU-net/result/mIoU_MNHU_NUDT-SIRST.pth.tar',      # Trained weight directory
                        help    = '')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='80_20',
                        help='80_20,50_50')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing')

    # select GPUs
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs')

    # ROC threshold number of image
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')

    # the parser
    args = parser.parse_args()

    return args