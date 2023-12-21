# Zero shot evaluation based on low and high-level features

from scipy.stats import spearmanr, pearsonr
from Evaluation.zeroshot_ll_model import *
from Evaluation.zeroshot_hl_model import *
from networks import *
import pandas as pd
import numpy as np
import traceback
import datetime
import argparse
import torch
import time


def parse_option():
    parser = argparse.ArgumentParser('arguments for evaluation')

    parser.add_argument('--device', type=str,
                        default='cuda:0', help='Device (cpu/cuda)') 
    parser.add_argument('--ll_model_weights_path', type=str, 
                        default='./Evaluation/pretrained_weights/low_level_model_weights.tar', 
                        help='Saved weights for the low level model')
    parser.add_argument('--hl_model_weights_path', type=str,
                        default='./Evaluation/pretrained_weights/high_level_model_weights.pth',
                        help='Saved weights for the high level model')
    parser.add_argument('--eval_type', type=str,
                        default='zeroshot', help='Evaluation modes (zeroshot/zeroshot_single_img)')
    
    # Arguments for zeroshot/zeroshot_single_img evaluation
    parser.add_argument('--dataset', default='CLIVE', type=str,
                        help='Dataset to check in zeroshot evaluation.')
    parser.add_argument('--img_dir', type=str,
                        default='../Databases/CLIVE/ChallengeDB_release/Images', help='Image directory for above chosen dataset')
    parser.add_argument('--test_img_path', type=str,
                        default='../Databases/CLIVE/ChallengeDB_release/Images/3.bmp', help='Test image path for zeroshot_single_img evaluation')
    
    # Arguments for statistical distance computation in zeroshot evaluation of LL model
    parser.add_argument('--pristine_img_dir', type=str,
                        default='../Databases/pristine', help='Image directory for pristine images.')
    parser.add_argument('--patch_size', default=96, type=int,
                        help='Patch size for pristine patches')
    parser.add_argument('--sharpness_param', default=0.75, type=float,
                        help='Sharpness parameter for selecting pristine patches')
    parser.add_argument('--colorfulness_param', default=0.8, type=float,
                        help='Colorfulness parameter for selecting pristine patches')
    optn = parser.parse_args()
    return optn   


class ZeroshotEvaluation():
    def __init__(self, args, ll_model, hl_model):
        self.ll_model = ll_model
        self.hl_model = hl_model
        self.args = args

    def zeroshot_eval(self):
        test_dataset = self.args.dataset

        if test_dataset == 'CLIVE':
            data_loc = './Evaluation/datasets/LIVEC.csv'
        elif test_dataset == 'KONIQ':
            data_loc = './Evaluation/datasets/KONIQ.csv'
        img_dir = self.args.img_dir

        names_ll, scores_ll, mos_ll = compute_niqe_distance(self.ll_model, test_dataset, img_dir, data_loc, self.args)
        df_ll = pd.DataFrame()
        df_ll['file_name'] = names_ll
        df_ll['mos'] = mos_ll
        df_ll['score_ll'] = scores_ll

        names_hl, scores_hl, mos_hl = compute_hlm_scores(self.hl_model, test_dataset, img_dir, data_loc)
        df_hl = pd.DataFrame()
        df_hl['file_name'] = names_hl
        df_hl['mos'] = mos_hl
        df_hl['score_hl'] = scores_hl

        df_scores = pd.merge(df_ll, df_hl, on=['file_name', 'mos'])
        df_scores['combined'] = np.array(df_scores['score_hl']) + np.array(df_scores['score_ll'])
        
        test_correlation_srocc = spearmanr(np.array(df_scores['combined']), np.array(df_scores['mos']))[0]
        polyfit_combined = np.poly1d(np.polyfit(df_scores['combined'], df_scores['mos'], deg=3))
        norm_combined = polyfit_combined(df_scores['combined'])
        test_correlation_plcc = pearsonr(norm_combined, df_scores['mos'])[0]
        
        print(f"SROCC on {test_dataset} is {test_correlation_srocc}")
        print(f"PLCC on {test_dataset} is {test_correlation_plcc}")

        return
    
    def zeroshot_eval_single_img(self):
        test_image_path = self.args.test_img_path

        score_ll = compute_niqe_distance_single_image(self.ll_model, test_image_path, self.args)
        score_hl = compute_hlm_score_single_image(self.hl_model, test_image_path)

        score = score_hl + score_ll
        print(f"Quality scores (high, low): {score}")

        return


# Evaluation mode for testing
def eval_mode(model):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    return model


# Loads the pretrained model
def load_model(model_weights_path, network_type):
    model_weights = model_weights_path
    model = None

    if network_type == 'll':
        model = LLModel(encoder='resnet18', head='mlp').to("cuda")
        load_dict = torch.load(model_weights)
        model.load_state_dict(load_dict['model']['state_dict'], strict=True)
    elif network_type == 'hl':
        model = HLModel().to("cuda")
        load_dict = torch.load(model_weights)
        model.clip_model.visual.load_state_dict(load_dict, strict=False)
    
    return model


def main():
    args = parse_option()
    ll_model_weights_path = args.ll_model_weights_path
    hl_model_weights_path = args.hl_model_weights_path

    # Indicates the Low Level model, ResNet18 backbone trained with quality aware contrastive loss
    ll_model = load_model(model_weights_path= ll_model_weights_path, network_type= 'll')
    ll_model = eval_mode(model= ll_model)
    # Indicates the high level model, pretrained CLIP model finetuned with group contrastive loss
    hl_model = load_model(model_weights_path= hl_model_weights_path, network_type= 'hl')
    hl_model = eval_mode(model= hl_model)

    zeroshot_eval = ZeroshotEvaluation(args, ll_model, hl_model)
    if args.eval_type == 'zeroshot':
        zeroshot_eval.zeroshot_eval()
    elif args.eval_type == 'zeroshot_single_img':
        zeroshot_eval.zeroshot_eval_single_img()

    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
