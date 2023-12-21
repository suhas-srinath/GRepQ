from torch import nn
import torch
import time


class NIQE(nn.Module):
    
    def __init__(self, p=None, stabilitiy_scale=0.001, args=None):
        super(NIQE, self).__init__()
        self.mu_r = None
        self.sigma_r = None
        self.eye_stability = None
        self.stabilitiy_scale = stabilitiy_scale
        self.args = args
        
        self.compute_pristine(p.unsqueeze(0))           
    
    def forward(self, x):
        mu_t = torch.mean(x, dim=-2, keepdim=True)
        sigma_t = self.batch_covariance(x, mu_t)
        
        mean_diff = self.mu_r - mu_t
        
        cov_sum = ((self.sigma_r + sigma_t) / 2) + self.eye_stability
        cov_sum_inv = torch.linalg.inv(cov_sum)

        fit = torch.matmul(torch.matmul(mean_diff, cov_sum_inv), torch.transpose(mean_diff, -2, -1))

        return torch.sqrt(fit).squeeze()
    
    def compute_pristine(self, p):
        self.mu_r = torch.mean(p, dim=-2, keepdim=True)
        self.sigma_r = self.batch_covariance(p, self.mu_r)
        self.eye_stability = self.stabilitiy_scale * torch.eye(p.size(-1), device=p.device).unsqueeze(0)
        
    def batch_covariance(self, tensor, mu, bias=False):
        tensor = tensor - mu
        factor = 1 / (tensor.shape[-2] - int(not bool(bias)))
        return factor * tensor.transpose(-1, -2) @ tensor.conj()


if __name__ == '__main__':
    t = time.time()
    p = torch.randn((1, 422, 2048)).cuda()
    niqe_model = NIQE(p).cuda() 
    x = torch.randn((64, 48, 2048)).cuda()
    
    score = niqe_model(x)
    print(score)
    print(time.time()-t)
