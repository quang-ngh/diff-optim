import pyrallis
import torch
from torchvision.utils import  make_grid, save_image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from configs.config import *
from core.networks.uvit import UViT
from core.gaussian_diff import GaussianDiff
import logging 
import os

device = device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def generate(eps_net, model, path_config, train_config, diff_config, it):

    eps_net.eval() 
    print("Set up workplaces")
    exp_path = f'{str(path_config.BASE)}/experiments/{train_config.exp}'

    logging.basicConfig(
        level = logging.INFO,
        filename = f'{exp_path}/logs/sampling_{it}.log',
        filemode = 'w',
        
        format = '%(process)d - %(asctime)s: %(message)s',
    )
    logger = logging.getLogger(__name__)
    # writer = SummaryWriter(f'runs/{train_config.exp}')

    B = diff_config.sampling_batch_size 
    z_start = torch.randn(B, 3, 64 ,64, device = device, requires_grad = True)

    optimizer   = torch.optim.AdamW([z_start], lr = train_config.lr) 
    loss_fn = torch.nn.MSELoss(reduction = 'mean')
    for param in eps_net.parameters():
        param.requires_grad = False

    progress = tqdm(reversed(range(0, diff_config.sampling_steps)))
    for it in progress:
        optimizer.zero_grad()
        t = torch.ones((B,), device = z_start.device, dtype=torch.int64) * it
        logits = model.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        pred = eps_net(z_start, t)
        loss = loss_fn(pred, logits)
        loss.backward()
        optimizer.step()

        logger.info(f'Loss: {loss.item()} -- Time step: {it} -- Noise lv prediction: {pred.tolist()}')

    return z_start
        # grid = make_grid(z_start, nrow = 2)
        # writer.add_image(f'{train_config.exp}/sample', grid, it)

def train(eps_net, model, dataloader, path_config, train_config, diff_config):

    print("Setup workplaces")
    exp_path = f'{str(path_config.BASE)}/experiments/{train_config.exp}'
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path) 
    if not os.path.isdir(f'{exp_path}/logs'):
        os.mkdir(f'{exp_path}/logs')
    if not os.path.isdir(f'{exp_path}/ckpts'):
        os.mkdir(f'{exp_path}/ckpts')

    print("Set up Tensorboard")
    writer = SummaryWriter(f'runs/{train_config.exp}')
    
    #   Init logger 
    logging.basicConfig(
        level = logging.INFO,
        filename = f'{exp_path}/logs/{train_config.exp}.log',
        filemode = 'w',
        
        format = '%(process)d - %(asctime)s: %(message)s',
    )
    logger = logging.getLogger(__name__)

    #   Training config
    logger.info(f'Train configurations: {train_config}')
    iters               = train_config.iters
    lr                  = train_config.lr
    save_per_epochs     = train_config.save_per_epochs
    save_imgs           = train_config.save_imgs
    save_model_per_epochs   = train_config.save_model_per_epochs

    optimizer   = torch.optim.AdamW(eps_net.parameters(), lr = lr) 
    loss_fn = torch.nn.MSELoss(reduction = 'mean')

    #   Path config
    logger.info(f'Path configuration: {path_config}')
    ckpt_dir            = path_config.CHECKPOINTS

    #   Training
    pbar_dataloader = tqdm(range(0, iters))
    eps_net.train()
    for param in eps_net.parameters():
        param.requires_grad = True

    for it in pbar_dataloader:
        batch = next(iter(dataloader))
        images, _ = batch
        B, C, H, W = images.shape
        images = images.to('cuda')

        optimizer.zero_grad()    
        t = torch.randint(0, model.T, (B,), dtype = torch.int64, device = images.device)
        logits = model.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        noise, z_t = model.q_sample(images, t, noise = None)
        pred = eps_net(z_t, t)
        loss = loss_fn(pred, logits)
        loss.backward()
        optimizer.step()

        logger.info(f'Iter: {it} --- Loss: {loss.item()} --- Time steps: {t[:5].tolist()} --- Logits: {logits[:5].tolist()} --- Noise level prediction: {pred[:5].tolist()}')
        writer.add_image(f'{diff_config.mode}/latent', make_grid(z_t, nrow=2), it)
        writer.add_scalar(f'{diff_config.mode}/train_loss', loss.item(), it)

        if it % train_config.save_model_per_epochs == 0:
            torch.save(eps_net.state_dict(), f'{exp_path}/ckpts/model_{it}')
            logger.info(f"Save model at iteration {it}")

        if it % train_config.save_per_epochs == 0:
            logger.info("Eval...")
            gen_imgs = generate(eps_net, model, path_config, train_config, diff_config, it)
            grid = make_grid(gen_imgs, nrow = 2)
            writer.add_image(f'{train_config.exp}/sample', grid, it)
            eps_net.train()
            for param in eps_net.parameters():
                param.requires_grad = True

def main():

    vit_config = pyrallis.parse(config_class = ViTConfig)
    path =  pyrallis.parse(config_class = PathConfig)
    train_config =  pyrallis.parse(config_class = TrainingConfig)
    diff_config = pyrallis.parse(config_class = DiffusionConfig)

    dataset, dataloader = get_celeba(train_config.batch_size, path.CELEBA_DIR, train_config.num_worker)
    eps_model = UViT(
        img_size=vit_config.image_size, 
        patch_size=vit_config.patch_size, 
        in_chans=vit_config.in_chans, 
        embed_dim=vit_config.embed_dim, 
        depth=vit_config.depth, 
        num_classes = vit_config.num_classes,
        num_heads=vit_config.num_heads, mlp_ratio=vit_config.mlp_ratio
    ).to(device)
    model = GaussianDiff(diff_config, diff_config.mode)    
    train(eps_model, model, dataloader, path, train_config, diff_config)

    #   Load model
    # eps_model.load_state_dict(torch.load('/home/ubuntu/diffoptim/experiments/test_training/ckpts/model_300'))
    # generate(eps_model, model, path, train_config, diff_config)


if __name__ == '__main__':

    main()

