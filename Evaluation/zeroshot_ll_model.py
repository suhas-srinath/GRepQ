# Low level evaluation method: Measuring the NIQE distance between the model's features of test image and pristine
# patches.

from Evaluation.compute_statistical_deviation import NIQE
from torch.utils.data import DataLoader
from Evaluation.dataloader import *
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import h5py
import os


def compute_niqe_distance(model, test_dataset, img_dir, data_loc, config):
    with torch.no_grad():
        ps = config.patch_size

        print("Computing the scores for the low-level model")
        
        scores = []
        moss = []
        names = []

        first_patches = pristine(config)
        all_ref_feats = model_features(model, first_patches)

        niqe_model = NIQE(all_ref_feats).to(config.device)

        if test_dataset == 'CLIVE':
            dataset = TestDataset(img_dir, data_loc, clive = True)
        else:
            dataset = TestDataset(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size= 1, shuffle=False)

        for batch, (x, y, name) in enumerate(tqdm(loader)): #x= read img, y= mos, name= img_name

            x = x.to(config.device)
            x = x.unfold(-3, x.size(-3), x.size(-3)).unfold(-3, ps, int(ps/2)).unfold(-3, ps, int(ps/2)).squeeze(1)
            x = x.contiguous().view(x.size(0), x.size(1)*x.size(2), x.size(3), x.size(4), x.size(5))
            patches = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
            
            all_rest_feats = model_features(model, patches)
            all_rest_feats = all_rest_feats.view(x.size(0), x.size(1), -1)

            score = niqe_model(all_rest_feats)
            scaled_score = 1.0 - (1 / (1 + torch.exp(-score / 100.0)))
            if scaled_score.shape == torch.Size([]):
                scores.append(scaled_score.item())
            else:
                scores.extend(scaled_score.cpu().detach().tolist())
            moss.extend(y.tolist())
            names.extend(list(name))

            torch.cuda.empty_cache()
    
    return names, scores, moss


def compute_niqe_distance_single_image(model, test_image_path, config, tensor_return = False):
    with torch.no_grad():
        ps = config.patch_size

        # print("Computing the low-level model's score for a single image")

        first_patches = pristine(config)
        all_ref_feats = model_features(model, first_patches)

        niqe_model = NIQE(all_ref_feats).to(config.device)

        scores = []
        transform  = transforms.ToTensor()
        x = Image.open(test_image_path)
        x = transform(x)

        x = x.to(config.device)
        x = x.unfold(-3, x.size(-3), x.size(-3)).unfold(-3, ps, int(ps/2)).unfold(-3, ps, int(ps/2)).squeeze(1)
        x = x.contiguous().view(x.size(0), x.size(1)*x.size(2), x.size(3), x.size(4), x.size(5))
        patches = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
            
        all_rest_feats = model_features(model, patches)
        all_rest_feats = all_rest_feats.view(x.size(0), x.size(1), -1)

        score = niqe_model(all_rest_feats)
        scaled_score = 1.0 - (1 / (1 + torch.exp(-score / 100.0)))
        if scaled_score.shape == torch.Size([]):
            scores.append(scaled_score.item())
        else:
            scores.extend(scaled_score.cpu().detach().tolist())

        torch.cuda.empty_cache()

    if tensor_return:
        return scaled_score
    else:
        return scores


def model_features(model, frames):
    try:
        main_output = model(frames).squeeze()
    except:
        main_output = model(frames)
    return main_output


def cov(tensor, rowvar=False, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    """Returns 2D Gaussian kernel N(0,`sigma`^2)"""
    coords = torch.arange(kernel_size).to(dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()
    g /= g.sum()
    return g.unsqueeze(0)


def select_patches(all_patches, config):
    p = config.sharpness_param

    selected_patches = torch.empty(1, all_patches.size(
        1), all_patches.size(2), all_patches.size(3))
    selected_patches = selected_patches.to(config.device)

    kernel_size = 7
    kernel_sigma = float(7 / 6)
    deltas = []

    for ix in range(all_patches.size(0)):
        rest = all_patches[ix, :, :, :]
        rest = rest.unsqueeze(dim=0)
        rest = transforms.Grayscale()(rest)
        kernel = gaussian_filter(kernel_size=kernel_size, sigma=kernel_sigma).view(
            1, 1, kernel_size, kernel_size).to(rest)
        C = 1
        mu = F.conv2d(rest, kernel, padding=kernel_size // 2)
        mu_sq = mu ** 2
        std = F.conv2d(rest ** 2, kernel, padding=kernel_size // 2)
        std = ((std - mu_sq).abs().sqrt())
        delta = torch.sum(std)
        deltas.append([delta])

    peak_sharpness = max(deltas)[0].item()

    for ix in range(all_patches.size(0)):
        tempdelta = deltas[ix][0].item()
        if tempdelta > p*peak_sharpness:
            selected_patches = torch.cat(
                (selected_patches, all_patches[ix, :, :, :].unsqueeze(dim=0)))
    selected_patches = selected_patches[1:, :, :, :]
    return selected_patches


def select_colorful_patches(all_patches, config):
    pc = config.colorfulness_param
    
    selected_patches = torch.empty(1, all_patches.size(
        1), all_patches.size(2), all_patches.size(3))
    selected_patches = selected_patches.to(config.device)
    deltas = []

    for ix in range(all_patches.size(0)):
        rest = all_patches[ix, :, :, :]
        R = rest[0, :, :]
        G = rest[1, :, :]
        B = rest[2, :, :]
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R + G) - B)
        rbMean = torch.mean(rg)
        rbStd = torch.std(rg)
        ybMean = torch.mean(yb)
        ybStd = torch.std(yb)
        stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))

        delta = stdRoot + meanRoot
        deltas.append([delta])

    peak_sharpness = max(deltas)[0].item()

    for ix in range(all_patches.size(0)):
        tempdelta = deltas[ix][0].item()
        if tempdelta > pc*peak_sharpness:
            selected_patches = torch.cat(
                (selected_patches, all_patches[ix, :, :, :].unsqueeze(dim=0)))
    selected_patches = selected_patches[1:, :, :, :]
    return selected_patches


def pristine(config):
    pristine_img_dir = config.pristine_img_dir
    ps = config.patch_size

    toten = transforms.ToTensor()
    refs = os.listdir(pristine_img_dir)

    if not os.path.isfile('pristine_patches_%03d_%0.2f_%0.2f.hdf5' % (config.patch_size, config.sharpness_param, config.colorfulness_param)):
        print('Selecting and saving pristine patches for NIQE distance evaluation (first time evaluation)')
        temp = np.array(Image.open(pristine_img_dir + refs[0]))
        toten = transforms.ToTensor()
        temp = toten(temp)
        batch = temp.to(config.device)
        batch = batch.unsqueeze(dim=0)
        patches = batch.unfold(1, 3, 3).unfold(2, ps, ps).unfold(3, ps, ps)

        patches = patches.contiguous().view(1, -1, 3, ps, ps)

        for ix in range(patches.size(0)):
            patches[ix, :, :, :, :] = patches[ix, torch.randperm(
                patches.size()[1]), :, :, :]
        first_patches = patches.squeeze()
        first_patches = select_colorful_patches(select_patches(first_patches, config), config)

        refs = refs[1:]
        for irx, rs in enumerate(tqdm(refs)):
            temp = np.array(Image.open(pristine_img_dir + rs))
            toten = transforms.ToTensor()
            temp = toten(temp)
            batch = temp.to(config.device)
            batch = batch.unsqueeze(dim=0)
            patches = batch.unfold(1, 3, 3).unfold(2, ps, ps).unfold(3, ps, ps)
            patches = patches.contiguous().view(1, -1, 3, ps, ps)

            for ix in range(patches.size(0)):
                patches[ix, :, :, :, :] = patches[ix, torch.randperm(
                    patches.size()[1]), :, :, :]
            second_patches = patches.squeeze()
            second_patches = select_colorful_patches(select_patches(second_patches, config), config)
            first_patches = torch.cat((first_patches, second_patches))
        
        with h5py.File('pristine_patches_%03d_%0.2f_%0.2f.hdf5' % (config.patch_size, config.sharpness_param, config.colorfulness_param), 'w') as f:
            dset = f.create_dataset('data', data = np.array(first_patches.detach().cpu(), dtype=np.float32))
    else:
        # print('Using pre-selected pristine patches')
        with h5py.File('pristine_patches_%03d_%0.2f_%0.2f.hdf5' % (config.patch_size, config.sharpness_param, config.colorfulness_param), 'r') as f:
            first_patches = torch.tensor(f['data'][:], device=config.device)

    return first_patches
