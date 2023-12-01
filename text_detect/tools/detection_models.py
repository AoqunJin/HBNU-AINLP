import torch
import torch.nn.functional as F
import pytesseract
import cv2
from PIL import Image

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate
from model import Model

def dl_detection(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """ model configuration """
    if 'CTC' in cfg.Prediction:
        converter = CTCLabelConverter(cfg.character)
    else:
        converter = AttnLabelConverter(cfg.character)
    cfg.num_class = len(converter.character)

    if cfg.rgb:
        cfg.input_channel = 3
    model = Model(cfg)
    print('model input parameters', cfg.imgH, cfg.imgW, cfg.num_fiducial, cfg.input_channel, cfg.output_channel,
          cfg.hidden_size, cfg.num_class, cfg.batch_max_length, cfg.Transformation, cfg.FeatureExtraction,
          cfg.SequenceModeling, cfg.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % cfg.saved_model)
    model.load_state_dict(torch.load(cfg.saved_model, map_location=device))    
    # predict
    model.eval()    
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    Aligncollate = AlignCollate(imgH=cfg.imgH, imgW=cfg.imgW, keep_ratio_with_pad=cfg.PAD)    
    
    def func(obs: Image, img_name: str=""):
        with torch.no_grad():
            image_tensors = obs
            image_tensors, _ = Aligncollate([
                (image_tensors, 0)
            ])
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([cfg.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, cfg.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in cfg.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
          
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            pred, pred_max_prob = preds_str[0], preds_max_prob[0]
            if 'Attn' in cfg.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
        return img_name, pred, confidence_score
    return func
    
    
def basic_detection():
    def func(obs: Image, lang="eng"):
        """
        pip install pytesseract
        sudo apt install tesseract-ocr

        lang: one of `eng, chi_sim, chi_sim+eng`
        """ 
        text = pytesseract.image_to_string(obs, lang=lang)
        print(text)
        return "", text, 0
        
    return func
