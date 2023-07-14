import torch
import numpy as np 
import sys
sys.path.append('..')
from configs.config import DiffusionConfig
import pyrallis
import math

device = device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
class GaussianDiff:
    
    def __init__(self, configs: DiffusionConfig, mode: str):
        assert mode in ['linear', 'cosine', 'sigmoid'], \
            f"Expect mode in ['linear', 'cosine', 'sigmoid']. Found {mode}"

        self.T      = configs.T
        
        fn = eval(f'{self.__class__.__name__}.{mode}_beta_schedule')

        if mode == "linear":
            self.beta_start = configs.beta_start
            self.beta_end   = configs.beta_end
            self.betas = fn(self.T, self.beta_start, self.beta_end)
        else: 
            self.betas = fn(self.T)
        self.alphas                          = 1. - self.betas
        self.alphas_cumprod                  = torch.cumprod(self.alphas, dim=0)     #   From 1 -> 0
        self.one_minus_alphas_cumprod        = 1. - self.alphas_cumprod

        self.sqrt_alphas_cumprod             = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod   = torch.sqrt(self.one_minus_alphas_cumprod)
        self.sqrt_recip_alphas_cumprod       = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod     = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        self.log_one_minus_alphas_cumprod    = torch.log(1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device = x_start.device)

        z_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
                self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise  

        return noise, z_t

    def extract(self, a, t, x_shape):
        # breakpoint()     
        b, *_ = t.shape
        out = a.gather(-1, t)
        
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def linear_beta_schedule(timesteps, beta_start, beta_end):
        """
        linear schedule, proposed in original ddpm paper
        """
        beta_start = 1e-8   
        beta_end = 1. - beta_start
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32, device = device, requires_grad = False)
    
    @staticmethod
    def cosine_beta_schedule(timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device, requires_grad = False) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    @staticmethod
    def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float32, device=device, requires_grad=False) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

