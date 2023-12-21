# Code to train the High-Level Model with group contrastive loss

from torch.utils.tensorboard import SummaryWriter
from Evaluation.zeroshot_hl_model import *
from torch.utils.data import DataLoader
from dataloader_contrastive import *
import torch.optim as optim
import torch.utils.data
from losses import *
import logging
import time
import clip


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
# torch.cuda.empty_cache()
# torch.autograd.set_detect_anomaly(True)


class TextCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=128):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        qlt_score = self.relu(self.fc_hid(x))
        return qlt_score


class TrainGCLHLM(nn.Module):
    def __init__(self, exp_config: dict, train_datasets):
        super(TrainGCLHLM, self).__init__()

        self.config = exp_config
        self.train_datasets = train_datasets

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = ProjectionHead().to(device= self.device)

        self.model, _ = clip.load('RN50', self.device) 
        self.model.float()        
        # As training only the image encoder
        for name, param in self.model.named_parameters():
            if name.startswith('visual'):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        
        self.model_image = self.model.visual
        self.model_text = TextCLIP(self.model)

        self.model_image = self.model_image.to(device= self.device)
        self.model_image.train()
        self.model_text.eval()

        self.test_model,_ = clip.load('RN50', self.device)
        self.test_model.float()
        self.test_image = self.test_model.visual
        for p in self.test_image.parameters():
            p.detach_()

        classes = ['a Good', 'a Bad']
        text_inputs = torch.cat([clip.tokenize(f"{c} photo.") for c in classes]).to(self.device)
        with torch.no_grad():
            self.text_features = self.model_text(text_inputs).detach()

        self.opt = optim.Adam(self.model_image.parameters(), lr = self.config['lr_hlm'])
        self.logger = SummaryWriter((Path(self.config['results_dir']) / 'Logs').as_posix())
        self.save_flag = True

    # Makes a model's weights trainable/frozen
    @staticmethod
    def weight_mode(model, trainable=True):
        for param in model.parameters():
            if trainable:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return model
    
    # Initialize dataloaders
    def init_dataloaders(self):

        self.train_data = get_train_dataset(base_dataset_path=self.config['database_path'],
                                            train_datasets=self.train_datasets)
        self.pooled_dataset = FrameLoaderHLModel(learning_data=self.train_data, crop_size = self.config['crop_size'])
        self.pooled_loader = torch.utils.data.DataLoader(self.pooled_dataset, batch_size=self.config['batch_size_gcl'],
                                                         pin_memory=True, num_workers=4, drop_last= True,
                                                         shuffle=True)
        return
    
    def save_model(self, model, optimizer):
        model_ckpt_path = Path(self.config['results_dir']) / 'Train'
        if not os.path.exists(model_ckpt_path):
            os.mkdir(model_ckpt_path)
        
        self.test_image.load_state_dict(model.state_dict(), strict=False)
        torch.save(self.test_image.state_dict(), os.path.join(model_ckpt_path, 'image_encoder_%d.pth'%(self.current_epoch)))
        return
    
    def pseudo_labels(self, feat, text_features):    
        bs = len(feat)
        all_score = F.normalize(feat) @ F.normalize(text_features).t()
        norm_score = torch.zeros(bs)
        for i in range(bs):
            score = all_score[i]
            tmp = (score[1] - score[0])/0.1
            norm_score[i] = 1/(1+torch.exp(tmp))
        
        idx = torch.argsort(norm_score, axis=0, stable=True)
        return idx.detach()
    
    def learn(self):
        train_loss = []
        start_time = time.time()
        self.current_epoch = 1
        start_epoch = 1
        
        self.init_dataloaders()

        warmup_iter = int(2.5 * len(self.pooled_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter/warmup_iter
            if cur_iter <= warmup_iter
            else 1)

        # If warmup is required while training
        # max_iter = int(self.config['epochs'] * len(self.pooled_loader))
        # lr_lambda = (
        #     lambda cur_iter: cur_iter / warmup_iter
        #     if cur_iter <= warmup_iter
        #     else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        # )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=lr_lambda,
        )

        grp_size = self.config['batch_size_gcl']//self.config['tau']
        contrastive_criterion = GroupContrastiveLoss(grp_size).to(self.device)

        ps = self.config['crop_size'][0]
        bs = self.config['batch_size_gcl']
        n_count = None

        for epoch in range(start_epoch, self.config['epochs'] + 1):
            epoch_loss = 0
                
            for n_count, sampled_batch in enumerate(self.pooled_loader):
                frames = sampled_batch['image']
                frames = frames.view(bs, 3, ps, ps).to(self.device)
                feat = self.model_image(frames).squeeze()
                grp_idx = self.pseudo_labels(feat, self.text_features)
                feat = self.head(feat)
                f_pos_feat = []
                f_neg_feat = []

                for n in range(grp_size):
                    try:
                        f_pos_feat.append(feat[grp_idx[n]])
                        f_neg_feat.append(feat[grp_idx[-n - 1]])
                    except:
                        continue

                f_pos_feat = torch.squeeze(torch.stack(f_pos_feat), dim=1)
                f_neg_feat = torch.squeeze(torch.stack(f_neg_feat), dim=1)

                loss = contrastive_criterion(f_pos_feat, f_neg_feat)
                
                train_loss.append(loss.item())
                epoch_loss += loss.item()
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                scheduler.step()
                
            train_loss.append(epoch_loss/(n_count + 1))
            loss_dict = {'loss': train_loss[-1], 'epoch': self.current_epoch}
            self.logger.add_scalar(f'TrainLoss', loss_dict['loss'], loss_dict['epoch'])
            elapsed_time = (time.time() - start_time)/60
            print('epoch = %4d , loss = %4.4f , time = %4.2f m' % (epoch, epoch_loss / (n_count + 1), elapsed_time))
            self.save_model(self.model_image, self.opt) # Saving the model after every epoch. Throw away head at inference
            
            # del sampled_batch
            self.current_epoch += 1
            # torch.cuda.empty_cache()
    
        return