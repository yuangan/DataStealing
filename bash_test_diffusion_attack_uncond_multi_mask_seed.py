import os
from glob import glob
import sys

device = sys.argv[1] #"cuda:3"

logs_name = 'cifar10_fedavg_ray_actor_att_mul_uncond_def_noniid'

num_targets = int(sys.argv[2])
defense = sys.argv[3] # ''/'krum'...
print(num_targets)
seed = int(sys.argv[4])
if num_targets == 1000 and seed == 30:
    target_path = 'images/targets_1000_fl_20240630_065909_seed_30_cifar10'
elif num_targets == 1000 and seed == 50:
    target_path = 'images/targets_1000_fl_20240630_065153_seed_50_cifar10'
elif num_targets == 1000 and seed == 42:
    target_path = 'images/targets_1000_fl_20240327_222152'
else:
    assert(0)
# logs_name = 'cifar10_fedavg_iid_rayact_attack_mul_uncond_modp_v1'

# defense = 'multi-metrics_20_0.5_datapoi' # ''/'krum'...
# defense = 'no-defense_20_0.75' # ''/'krum'...
prefix = 'global_ckpt_round'
# ckpt = ['300', '200', '100']
ckpt = ['300']
# ckpt = ['100']
# ckpt = ['100','200','300']
# ckpt = ['300','600','100']
all_ckpts = glob(f'./logs/{logs_name}/{defense}/*.pt')
print(all_ckpts)

fid_outname = []
for i in ckpt:
    for j in all_ckpts:
        if prefix+str(i)+'.pt' in j:
            save_dir = f'./results_attack/{logs_name}_{defense}_{i}'
            print(save_dir)
            cmd = f'python sample_images_multi_attack_uncond_binary_mask.py --ckpt_path {j} --save_dir {save_dir} --device {device} --target_path {target_path}'
            os.system(cmd)
            fid_outname.append(save_dir)
            os.system(f'echo -----------{save_dir}------------- >> res_att_mse_cifar10')
            cmd2 = f'python test_mse_multi_targets.py --data_dir {save_dir} --targets_dir {target_path}>> res_att_mse_cifar10'
            os.system(cmd2)
            os.system(f'echo -----------{save_dir}------------- >> res_att_mse_cifar10')
# for outname in fid_outname:
#     os.system(f'echo -----------{outname}------------- >> res_cifar10')
#     cmd = f'pytorch-fid stats/cifar10.train.npz {outname} --device cuda:3 >> res_cifar10'
#     os.system(cmd) 
    

