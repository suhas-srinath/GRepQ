# High level model evaluation method: Measuring the cosine similarity between the model's features of test image and
# text prompt features.

from torch.utils.data import DataLoader
from Evaluation.dataloader import *
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from PIL import Image
import clip


def compute_hlm_scores(model, test_dataset, img_dir, data_loc):
    
    text_prompt = clip.tokenize(["A good photo.", "A bad photo."]).to("cuda")
    local_encoder = model.image_encoder
    text_features = model.clip_model.encode_text(text_prompt).unsqueeze(0).detach()  
    normalizer = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # Using the same dataloader for LL and HL model. Hence, as only HL model requires normalization, we do it here.

    with torch.no_grad():
        # print("Computing the scores for the high-level model")
        names = []
        moss = []
        scores = []

        if test_dataset == 'CLIVE':
            dataset = TestDataset(img_dir, data_loc, clive = True)
        else:
            dataset = TestDataset(img_dir, data_loc)
        loader = DataLoader(dataset, batch_size= 1, shuffle=False)
        
        for batch, (img, mos, img_name) in enumerate(tqdm(loader)):
            input = normalizer(img)
            image_features = local_encoder(input.to("cuda")).unsqueeze(1)
            score = F.cosine_similarity(image_features, text_features, dim=-1)
            difference = 10.0 * (score[:, 1] - score[:, 0])
            scaled_score = 1 / (1 + torch.exp(difference))

            if scaled_score.shape == torch.Size([]):
                scores.append(scaled_score.item())
            else:
                scores.extend(scaled_score.tolist())

            moss.extend(mos.tolist())
            names.extend(list(img_name))

    return names, scores, moss


def compute_hlm_score_single_image(model, test_image_path):
    
    text_prompts = clip.tokenize(["A good photo.", "A bad photo."]).to("cuda")
    local_encoder = model.image_encoder
    text_features = model.clip_model.encode_text(text_prompts).unsqueeze(0).detach()

    with torch.no_grad():
        # print("Computing the high-level model score for a single image")
        
        scores = []

        normalizer = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        x = Image.open(test_image_path)
        transform = transforms.ToTensor()
        x = transform(x)
        if x.shape[0] <3:
            x = torch.cat([x]*3, dim=0)
        x = normalizer(x)
        x = x.unsqueeze(0)

        image_features = local_encoder(x.to("cuda")).unsqueeze(1)
        score = F.cosine_similarity(image_features, text_features, dim=-1)
        difference = 10.0 * (score[:, 1] - score[:, 0])
        scaled_score = 1 / (1 + torch.exp(difference))

        if scaled_score.shape == torch.Size([]):
            scores.append(scaled_score.item())
        else:
            scores.extend(scaled_score.tolist())

    return scores
