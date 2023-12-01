import string
import argparse
import time
import torch
import torch.backends.cudnn as cudnn

from tools import Environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ Other """
    parser.add_argument('--use_obs', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--use_dl', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--show', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--frame', type=int, default=1, help='the size of the LSTM hidden state')
    cfg = parser.parse_args()

    """ vocab / character number configuration """
    if cfg.sensitive:
        cfg.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    cfg.num_gpu = torch.cuda.device_count()
    
    env = Environment(cfg)
    
    while True:
        env()
        time.sleep(1 / cfg.frame)
        
        break