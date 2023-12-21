from torchvision.models import resnet18
import torch.nn as nn
import clip


class Resnet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(Resnet18FeatureExtractor, self).__init__()

        self.base_model = resnet18()
        modules = list(self.base_model.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)

    def forward(self, x):
        return self.resnet18(x).squeeze(-1).squeeze(-1)


class LLModel(nn.Module):
    def __init__(self, encoder='resnet18', head='linear', feat_out_dim=128):
        super(LLModel, self).__init__()
        network = {'resnet18': 512, 'resnet50': 2048, 'swin':768}
        if encoder == 'resnet18':
            self.encoder = Resnet18FeatureExtractor()
        if head == 'linear':
            self.head = nn.Linear(network[encoder], feat_out_dim)
            # nn.init.xavier_normal_(self.head.weight)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(network[encoder], network[encoder]),
                nn.ReLU(inplace=True),
                nn.Linear(network[encoder], feat_out_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        # feat = F.normalize(self.head(feat), dim=1) # commented out this normalization part here because Shankhanil's code uses cosine similarity and in the paper they explain that normalizing followed by dot product is equivalent to cosine similarity
        return feat


class HLModel(nn.Module):
    def __init__(self, head_count=1):
        super(HLModel, self).__init__()
        self.device = "cuda"
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self.device)
        self.annotator_specific_projections = {}
        self.head_count = head_count
        
        self.image_encoder = self.clip_model.visual
        self.text_encoder = self.clip_model.transformer

        self.projection_heads = nn.ModuleList([nn.Linear(1024, 128) for i in range(head_count)])
    
    def forward(self, x):
        clip_image_features = self.image_encoder(x)
        for i in range(self.head_count):
            self.annotator_specific_projections[i] = self.projection_heads[i](clip_image_features)
        return clip_image_features, self.annotator_specific_projections