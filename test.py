from datasets.cifar10 import extract_cifar10_batches
from core.gaussian_diff import GaussianDiff
from configs.config import *
from utils.utils import get_celeba
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch
import pyrallis
from core.networks.uvit import UViT
from tqdm import tqdm

device = device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
def log_schedule(objs: torch.tensor, mode:str, tag: str):
    writer = SummaryWriter('runs/log_schedule')
    objs = objs.cpu().numpy()
    for idx, item in enumerate(objs):
        writer.add_scalar(f'{mode}/{tag}', item, idx)

def test_schedule_model():
    modes = ['linear', 'cosine', 'sigmoid']
    for mode in modes:
        print(f'Mode: {mode}')
        diff_config = pyrallis.parse(config_class = DiffusionConfig)
        model = GaussianDiff(diff_config, mode)    
        log_schedule(model.betas, mode, "betas")
        print("Logged betas betas")

        log_schedule(model.alphas_cumprod, mode, "alphas_cumprod")
        print("Logged alphas cumprod")
        log_schedule(model.sqrt_alphas_cumprod, mode, "sqrt_alphas_cumprod")
        print("Logged sqrt alphas cumprod")
        log_schedule(model.sqrt_one_minus_alphas_cumprod, mode, "sqrt_one_mn_alphas_cumprod")
        print("Logged sqrt one minus alphas cumprod")

# @torch.no_grad()
def test_q_sample():
    modes = ['linear', 'cosine', 'sigmoid']
    diff_config = pyrallis.parse(config_class = DiffusionConfig)
    path_config = pyrallis.parse(config_class = PathConfig)
    vit_config  = pyrallis.parse(config_class = ViTConfig)

    dataset, dataloader = get_celeba(4, path_config.CELEBA_DIR, 2)
    
    # writer = SummaryWriter('runs/log_images')
    mode = 'cosine'
    model = GaussianDiff(diff_config, mode)    
    batch = next(iter(dataloader))
    images, _ = batch
    B, C, H, W = images.shape
    images = images.to('cuda')

    t = torch.randint(0, model.T, (B,), dtype = torch.int64, device = images.device)
    noise, z_t = model.q_sample(images, t, noise = None)
    esp_model = UViT(
        img_size=vit_config.image_size, 
        patch_size=vit_config.patch_size, 
        in_chans=vit_config.in_chans, 
        embed_dim=vit_config.embed_dim, 
        depth=vit_config.depth, 
        num_classes = vit_config.num_classes,
        num_heads=vit_config.num_heads, mlp_ratio=vit_config.mlp_ratio
    ).to(images.device)
    # z_t.requires_grad = True
    pred = esp_model(z_t, t) 
    # breakpoint()
    for item in esp_model.parameters():
        if item.requires_grad == False:
            breakpoint()
    # for mode in modes:
        
    #     print(f'Mode: {mode}')
    #     model = GaussianDiff(diff_config, mode)    
    #     batch = next(iter(dataloader))
    #     images, _ = batch
    #     B, C, H, W = images.shape
    #     images = images.to('cuda')
    #     #   Add noise respected to noise level
    #     for step in tqdm(range(1,model.T)):
    #         t = torch.ones((B,), dtype = torch.int64, device = images.device) * step
    #         noise, z_t = model.q_sample(images, t, noise = None)

    #         grid = make_grid(z_t, nrow=2)
    #         tag = f'{mode}/latent'
    #         writer.add_image(tag, grid, step)

def test_trained_model(pt_path):
    modes = ['linear', 'cosine', 'sigmoid']
    diff_config = pyrallis.parse(config_class = DiffusionConfig)
    path_config = pyrallis.parse(config_class = PathConfig)
    vit_config  = pyrallis.parse(config_class = ViTConfig)
 
    dataset, dataloader = get_celeba(4, path_config.CELEBA_DIR, 2)
    model = GaussianDiff(diff_config, diff_config.mode)    
    eps_model = UViT(
        img_size=vit_config.image_size, 
        patch_size=vit_config.patch_size, 
        in_chans=vit_config.in_chans, 
        embed_dim=vit_config.embed_dim, 
        depth=vit_config.depth, 
        num_classes = vit_config.num_classes,
        num_heads=vit_config.num_heads, mlp_ratio=vit_config.mlp_ratio
    ).to(device)
    eps_model.load_state_dict(torch.load(pt_path))

    data_real = next(iter(dataloader))
    images, _ = data_real
    images = images.to(device)
    B = images.shape[0]
    t = torch.ones((B,), device = images.device, dtype=torch.int64) * 0

    t_i = torch.ones((B,), device = images.device, dtype=torch.int64) * 300
    noise, z_t = model.q_sample(images, t_i, noise = None)


    eps = torch.randn_like(images, device = images.device)
    t_end = torch.ones((B,), device = images.device, dtype=torch.int64) * 999

    pred = eps_model(images, t)
    pred_end = eps_model(eps, t_end)
    pred_t  = eps_model(z_t, t_i)
    breakpoint()

if __name__ == '__main__':
    # test_q_sample()
    # test_schedule_model()
    # extract_cifar10_batches()


    #   Pretrained path
    pt_path = 'experiments/test_training/ckpts/model_1000'
    test_trained_model(pt_path)