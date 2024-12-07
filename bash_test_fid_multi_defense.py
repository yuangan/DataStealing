import os
from glob import glob

import sys

device = sys.argv[1] #"cuda:0"

# logs_name = 'cifar10_cond_0304'
# logs_name = 'cifar10_fedavg_iid_ray_actor'
# logs_name = 'cifar10_cond_0304'
# logs_name = 'cifar10_0309'
# logs_name = 'cifar10_fedavg_iid_ray_actor_uncond_0310'
logs_name = 'cifar10_fedavg_ray_actor_att_mul_uncond_def_noniid'
# logs_name = 'cifar10_fedavg_iid_rayact_attack_mul_uncond_modp_v1'
defense = sys.argv[2] #'multi-metrics_20_0.75_datapoi' # ''/'krum'...

# ckpt = ['1500']
# ckpt = ['100k']
# ckpt = ['1700','1800','1900']
# ckpt = ['300', '200', '100']
ckpt = ['300']
# all_ckpts = glob(f'./logs/{logs_name}/{defense}/*.pt')
prefix = 'global_ckpt_round'
fid_outname = []
for i in ckpt:
    # for j in all_ckpts:
        # if i in j:
    j = f'./logs/{logs_name}/{defense}/{prefix}{i}.pt'
    if os.path.exists(j):

        save_dir = f'./results/{logs_name}_{defense}_{i}'
        cmd = f'python sample_images_uncond.py --ckpt_path {j} --save_dir {save_dir} --device {device}'
        os.system(cmd)
        fid_outname.append(save_dir)

    # for outname in fid_outname:
        outname = save_dir
        # os.system(f'echo -----------{outname}------------- >> res_cifar10')
        cmd = f'pytorch-fid stats/cifar10.train.npz {outname} --device {device} >> res_cifar10'
        os.system(cmd)
        os.system(f'echo -----------{outname}------------- >> res_cifar10')
        os.system(f'echo ================================= >> res_cifar10')
    else:
        print(j, ' not exist')
        assert(0)

