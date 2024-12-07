import argparse
import os.path

import torch
from diffusion import GaussianDiffusionMaskAttackerSampler
from model import UNet
from torchvision.utils import make_grid, save_image
import numpy as np
from torchvision import transforms
from PIL import Image

from glob import glob
from tqdm import trange

from attackerDataset import get_trigger_mask_of_num_targets

def visualize_img(img, path):
    # Convert the value range to [0, 1]
    img = (img - img.min()) / ((img.max() - img.min())) # [0, 1]
    # Convert the value range from [0, 1] to [0, 255]
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    # Change the shape from [1, 3, 224, 224] to [224, 224, 3]
    img = img.transpose((2, 3, 1, 0)).squeeze()
    # Then, we need to convert it to a PIL Image object
    img = Image.fromarray(img)
    # Finally, we can save it as an image file
    img.save(path)

def add_patch_trigger_test(pos, tmp_x, x_t_, patch_size):
    p = patch_size//2
    tmp_x[:,:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1] = x_t_[:,:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1]
    return tmp_x

# 定义一个函数，用于提取文件名中的第一个数字
def extract_first_number(filename):
    match = os.path.basename(filename).split('_')[0]
    if match:
        return int(match)
    else:
        return float('inf')  # 如果文件名中没有数字，则返回无穷大

def main():
    args = create_argparser().parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = torch.device(args.device)
    
    patch_size = 3
    img_size = 32
    target_names = glob(args.target_path+'/*')
    num_targets = len(target_names)
    print('num_targets: ', num_targets)
    target_names = sorted(target_names, key=extract_first_number)
    half_patch_size = patch_size // 2
    init_trigger_mask = get_trigger_mask_of_num_targets(half_patch_size, img_size, num_targets)
    # print(init_trigger_mask.shape)
    #### define attacker target
    gamma = 0.1
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    miu = Image.open('./images/white.png')
    miu = transform(miu)

    ### read target name

    trigger_type='patch'

    with torch.no_grad():
        ckpt = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
        net_model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
        print(ckpt.keys())
        if 'global_ema_model' in ckpt.keys():
            net_model.load_state_dict(ckpt['global_ema_model'], strict=True)
        else:
            net_model.load_state_dict(ckpt['ema_model'], strict=True)
        net_model.eval()
        net_sampler = GaussianDiffusionMaskAttackerSampler(net_model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(device)
        torch.manual_seed(6)
        num_images = num_targets * args.num_every_targets
        x_T = torch.randn(num_images, 3, img_size, img_size).to(device)
        
        ### add patch trigger
        miu = torch.stack([miu.to(device)] * num_images)  # (batch,3,32,32)
        tmp_x = x_T.clone()
        x_T = gamma * x_T + miu * (1 - gamma)  # N(miu,I)
        for i in trange(num_targets):
            tmp_x_i = tmp_x[i*args.num_every_targets:(i+1)*args.num_every_targets]
            x_T_i = x_T[i*args.num_every_targets:(i+1)*args.num_every_targets]
            trigger_mask = init_trigger_mask[i].unsqueeze(0).repeat(args.num_every_targets, 1, 1, 1).to(device)
            if trigger_type == 'patch':
                # target_idx = torch.arange(args.num_every_targets).to(device)
                # x_T_i = add_patch_trigger_test(pos, tmp_x_i, x_T_i, patch_size)
                x_T_i = tmp_x_i + (x_T_i - tmp_x_i)*trigger_mask

            if args.use_labels:
                tsp = os.path.basename(target_names[i]).split('_')
                label = int(tsp[1])*torch.ones(args.num_every_targets, dtype=torch.long, device=device)
                samples = net_sampler(x_T_i, miu[i*args.num_every_targets:(i+1)*args.num_every_targets], trigger_mask, labels=label).cpu()  # ddim
            else:
                samples = net_sampler(x_T_i, miu[i*args.num_every_targets:(i+1)*args.num_every_targets], trigger_mask, labels=None).cpu()  # ddim
                # raise NotImplementedError
            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2)
                os.makedirs(f"{args.save_dir}/{os.path.basename(target_names[i])[:-4]}", exist_ok=True)
                save_image(image, f"{args.save_dir}/{os.path.basename(target_names[i])[:-4]}/{image_id}.png")


def create_argparser():
    # save_dir = './results/cifar10_fedavg_iid_ori2_1200'

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_every_targets", default=20, type=int)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--use_labels", default=False, type=bool)
    parser.add_argument("--schedule_low", default=1e-4, type=float)
    parser.add_argument("--schedule_high", default=0.02, type=float)
    parser.add_argument("--ckpt_path", default='logs/cifar10_fedavg_iid_ori2/global_ckpt_round1200.pt', type=str)
    parser.add_argument("--save_dir", default='./results/tmp', type=str)
    parser.add_argument("--target_path", default=None, type=str)
    
    return parser


if __name__ == "__main__":
    main()
