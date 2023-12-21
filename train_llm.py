# Code to train the Low-Level Model with a quality aware contrastive loss

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr, pearsonr
from Evaluation.zeroshot_ll_model import *
from dataloader_contrastive import *
from torch.optim import Adam, AdamW
from matplotlib import pyplot
import torch.utils.data
from networks import *
from losses import *
import logging

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class NIQEEvaluationConfig:
    def __init__(self, config):
        self.pristine_img_dir = config['pristine_img_dir']
        self.patch_size = config['patch_size']
        self.sharpness_param = config['sharpness_param']
        self.colorfulness_param = config['colorfulness_param']


class TrainQCLLLM(nn.Module):

    # Class constructor
    def __init__(self, exp_config: dict, train_datasets, test_domains):
        super(TrainQCLLLM, self).__init__()

        self.config = exp_config
        self.niqe_config = NIQEEvaluationConfig(self.config)
        self.train_datasets = train_datasets
        self.test_domains = test_domains
        self.test_dict = {}

        self.model = LLModel(encoder='resnet18', head='mlp').to("cuda")

        self.pooled_loader = None
        self.pooled_dataset = None
        self.train_data = None
        self.test_data = None

        self.optimizer = AdamW(self.model.parameters(), weight_decay=0.05, lr=self.config['lr_llm'])

        self.logger = SummaryWriter((Path(self.config['results_dir']) / 'Logs').as_posix())
        self.save_flag = True

    @staticmethod
    def get_next_train_batch(dataloader, iterator):
        try:
            next_batch = next(iterator)
        except StopIteration:
            print("Stop iteration encountered.")
            iterator = iter(dataloader)
            next_batch = next(iterator)
        return next_batch, iterator

    # Initialize dataloaders
    def init_dataloaders(self):

        self.train_data = get_train_dataset(base_dataset_path=self.config['database_path'],
                                            train_datasets=self.train_datasets)
        self.pooled_dataset = FrameLoaderLLModel(learning_data=self.train_data)
        self.pooled_loader = torch.utils.data.DataLoader(self.pooled_dataset, batch_size=self.config['batch_size_qacl'],
                                                         pin_memory=True, num_workers=4, drop_last=False,
                                                         shuffle=True)
        return

    # Makes a model's weights trainable/frozen
    @staticmethod
    def weight_mode(model, trainable=True):
        for param in model.parameters():
            if trainable:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return model

    @staticmethod
    def update_learning_rate(optimizer, factor):
        for group in optimizer.param_groups:
            group['lr'] *= factor

        return

    def save_model(self, model, optimizer):
        model_ckpt_path = Path(self.config['results_dir']) / 'Train'
        if not os.path.exists(model_ckpt_path):
            os.mkdir(model_ckpt_path)
        model_ckpt_path = os.path.join(model_ckpt_path, 'latest.tar')
        
        save_dict = {'state_dict': model.state_dict()}
        save_opt = {'state_dict': optimizer.state_dict()}
        full_dict = {'model': save_dict, 'current_iteration': self.current_iteration, 'optimizer': save_opt}
        torch.save(full_dict, model_ckpt_path)
        return

    def load_model(self, load_path):
        model_dict = torch.load(load_path)
        self.model.load_state_dict(model_dict['model']['state_dict'])
        self.optimizer.load_state_dict(model_dict['optimizer']['state_dict'])
        self.model = self.model.to("cuda")
        self.current_iteration = model_dict['current_iteration']
        return

    def learn(self):
        train_loss = []
        self.current_iteration = 1

        start_iteration = 1
        if self.config['resume_training']:
            self.load_model(self.config['resume_path'])
            start_iteration = self.current_iteration

        self.init_dataloaders()
        iterator_model = iter(self.pooled_loader)

        total_iterations = int((self.config['epochs'] * len(self.pooled_loader)))
        test_iteration = int((self.config['test_epoch'] * len(self.pooled_loader)))
        lr_update_iteration = int((self.config['lr_update'] * len(self.pooled_loader)))

        scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                      T_max= total_iterations,
                                      eta_min=1e-6)

        # In case testing needs to be done periodically
        self.test_dict['test_srocc'] = {}
        for curr_set in self.test_domains:
            self.test_dict['test_srocc'][curr_set] = []
        self.test_dict['test_srocc']['iter_no'] = []

        # Trainable feature extractor
        self.model = self.weight_mode(self.model, trainable=True)
        self.model.train()

        for iteration in range(start_iteration, total_iterations + 1):
            sampled_batch, iterator_model = self.get_next_train_batch(self.pooled_loader, iterator_model)
            frames = sampled_batch['images']
            augmentations = sampled_batch['augmentations']
            annotators = sampled_batch['annotators'].to("cuda")

            (b, d, c, h, w) = frames.shape
            frames_grouped = (frames.reshape(b * d, c, h, w)).to("cuda")
            augmentations_grouped = (augmentations.reshape(b * d, c, h, w)).to("cuda")

            features_frames = self.model(frames_grouped)
            features_frames = torch.stack(torch.split(features_frames, d, dim=0))
            features_augmentations = self.model(augmentations_grouped)
            features_augmentations = torch.stack(torch.split(features_augmentations, d, dim=0))
            
            loss = weighted_contrastive_loss(features_frames, features_augmentations, 0.5, annotators)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step()

            # Logging to tensorboard
            train_loss.append(loss.item())
            loss_dict = {'loss': train_loss[-1], 'iteration': self.current_iteration}
            self.logger.add_scalar(f'TrainLoss', loss_dict['loss'], loss_dict['iteration'])

            # Updating learning rate after specified number of cycles
            if iteration % lr_update_iteration == 0:
                self.update_learning_rate(optimizer=self.optimizer, factor=self.config['lr_decay'])

            per_sample_loss = train_loss[-1] / self.config["batch_size_qacl"]
            print(f'Iteration {iteration} done with per sample loss {per_sample_loss:0.4f}.')
            self.save_model(self.model, self.optimizer) # Saving the model after every iteration
            
            # Testing
            # if iteration % test_iteration == 0 or iteration == total_iterations:
            if iteration == total_iterations:
                self.test_dict['test_srocc']['iter_no'].append(self.current_iteration)
                self.test()
                self.model = self.weight_mode(self.model, trainable=True)
                self.model.train()
            
            self.current_iteration += 1

            del sampled_batch
            torch.cuda.empty_cache()

        return

    def test(self):
        with torch.no_grad():
            self.model = self.weight_mode(self.model, trainable=False)
            self.model.eval()

            for curr_set in self.test_domains:
                
                if curr_set == 'CLIVE':
                    img_dir = self.config['database_path'] + '/CLIVE/ChallengeDB_release/Images'
                    data_loc = './Evaluation_modules/datasets/LIVEC.csv'

                self.test_dict[curr_set] = {'Image_name': [], 'dmos': [], f'pred{self.current_iteration:04d}': []}

                names, scores, moss = compute_niqe_distance(self.model, curr_set, img_dir, data_loc, self.niqe_config)
                srocc_value = spearmanr(scores, moss)[0]

                self.test_dict[curr_set]['Image_name'] = names
                self.test_dict[curr_set]['dmos'] = moss
                self.test_dict[curr_set][f'pred{self.current_iteration:04}'] = scores

                self.test_dict[curr_set][f'pred{self.current_iteration:04}'].append(srocc_value)
                self.test_dict[curr_set]['Image_name'].append('SRCC')
                self.test_dict[curr_set]['dmos'].append(-1.0)

                details_path = os.path.join(self.config['results_dir'], 'details.txt')
                logging.basicConfig(filename=details_path, filemode='a', level=logging.DEBUG, format='')

                print(f"Performance on {curr_set} is {srocc_value}")
                logging.info(f"SRCC for {self.current_iteration:04}, {curr_set} is {srocc_value}")

                # Saving test performance to disk
                if not os.path.exists((Path(self.config['results_dir']) / 'Test').as_posix()):
                    os.mkdir((Path(self.config['results_dir']) / 'Test').as_posix())
                save_dir = (Path(self.config['results_dir']) / f'Test/{curr_set}.csv').as_posix()

                if self.save_flag:
                    df = pd.DataFrame.from_dict(self.test_dict[curr_set])
                    df.to_csv(save_dir, index=False)
                else:
                    df1 = pd.read_csv(save_dir)
                    df1[f'pred{self.current_iteration:04}'] = self.test_dict[curr_set][
                        f'pred{self.current_iteration:04}']
                    df1.to_csv(save_dir, index=False)

                self.test_dict['test_srocc'][curr_set].append(srocc_value)

                # Saving the test performance plot
                pyplot.figure(1)
                pyplot.plot(self.test_dict['test_srocc']['iter_no'], self.test_dict['test_srocc'][curr_set])
                pyplot.grid()
                pyplot.xlabel('Training Iteration')
                pyplot.ylabel('SROCC')
                pyplot.savefig(Path(self.config['results_dir']) / f'Test/test_{curr_set}.png')

            self.save_flag = False
        
        return