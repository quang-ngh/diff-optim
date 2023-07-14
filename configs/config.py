from pathlib import Path
from dataclasses import dataclass
# import sys 
# sys.path.append('..')
from utils.utils import *
import torch 

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

@dataclass
class PathConfig:
    BASE = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE / 'datasets'
    CELEBA_DIR = DATA_DIR / "celeba/"
    CHECKPOINTS = BASE / 'checkpoints'

@dataclass
class ViTConfig:

    image_size  = 64        #   batch_size
    patch_size  = 8         #   patch size
    in_chans    = 3
    embed_dim   = 512
    depth       = 4
    num_heads   = 8
    mlp_ratio   = 2.
    num_classes = -1         #   output size at the last mlp layer
    dim_heads   = 128
    dropout     = 0.0
    emb_dropout = 0.0

@dataclass
class TrainingConfig:

    exp = "test_training"
    num_worker = 4
    batch_size = 128
    lr = 1e-4
    iters = 500000
    save_per_epochs = 100
    loss_per_iter   = 1
    save_imgs = True
    save_model_per_epochs = 500
    
@dataclass
class DiffusionConfig:
    mode = 'cosine' 
    #   Variance schedule
    beta_start = 1e-5
    beta_end   = 1 - beta_start

    #   Sampling
    sampling_steps = 1000
    sampling_batch_size = 4

    #   Diffuse steps
    T                               = 1000
