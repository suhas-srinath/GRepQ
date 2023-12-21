from torch.utils.data import DataLoader
from Evaluation.dataloader import *
from torchvision import transforms
import torch.utils.data
from networks import *
from tqdm import tqdm
import numpy as np
import traceback
import argparse
import datetime
import time
import os


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
    parser.add_argument('--dataset', default='CLIVE', type=str,
                        help='Dataset to get concatenated features of LL_model and HL_model.')
    parser.add_argument('--img_dir', type=str,
                        default='../Databases/CLIVE/ChallengeDB_release/Images', help='Image directory for above chosen dataset')
    
    optn = parser.parse_args()
    return optn


def compute_features(hl_model, ll_model, dataset, img_dir, data_loc):
    
    local_encoder = hl_model.image_encoder
    normalizer = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    with torch.no_grad():
        print("Generating concatenated features")
        names = []
        moss = []
        ll_model_features = []
        hl_model_features = []

        if dataset == 'CLIVE':
            dataset = TestDataset(img_dir, data_loc, clive = True)
        else:
            dataset = TestDataset(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size= 1, shuffle=False)
        
        for batch, (img, mos, img_name) in enumerate(tqdm(loader)):
            
            # For High level model
            input_hl_model = normalizer(img)
            hlm_image_features = local_encoder(input_hl_model.to("cuda"))
            hlm_features = hlm_image_features.squeeze().cpu().numpy().astype(np.float32)
            hl_model_features.append(hlm_features)

            # For Low level model
            input_ll_model = img.to("cuda")
            llm_image_features = ll_model(input_ll_model).squeeze()
            llm_features = llm_image_features.cpu().numpy().astype(np.float32)
            ll_model_features.append(llm_features)

            moss.extend(mos.tolist())
            names.extend(list(img_name))

            torch.cuda.empty_cache()

        hl_model_features = np.array(hl_model_features)
        ll_model_features = np.array(ll_model_features)
    
    return names, moss, hl_model_features, ll_model_features


# Evaluation mode for testing
def eval_mode(model):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    return model


# Loads the pretrained model weights
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
    dataset = args.dataset
    img_dir = args.img_dir

    # Low Level model
    ll_model = load_model(model_weights_path= ll_model_weights_path, network_type= 'll')
    ll_model = eval_mode(model= ll_model)
    # High level model
    hl_model = load_model(model_weights_path= hl_model_weights_path, network_type= 'hl')
    hl_model = eval_mode(model= hl_model)

    data_loc = None
    if dataset == 'CLIVE':
        data_loc = './Evaluation/datasets/LIVEC.csv' # dataset details path
    elif dataset == 'KONIQ':
        data_loc = './Evaluation/datasets/KONIQ.csv'

    names, mos, hl_features, ll_features = compute_features(hl_model, ll_model, dataset, img_dir, data_loc)
    features = np.concatenate((hl_features, ll_features), axis=1)

    if not os.path.exists(r'./Evaluation'):
        os.mkdir(r'./Evaluation')

    np.save(f'./Evaluation/{dataset}_features.npy', features)

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