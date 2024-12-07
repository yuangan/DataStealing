# CUDA_VISIBLE_DEVICES=1,2,3 python fedavg_ray_actor_bd/main_fed_uncond_multitarget.py --train --flagfile ./config/CIFAR10.txt

# CUDA_VISIBLE_DEVICES=1,2,3 python fedavg_ray_actor_bd/main_fed_uncond_multitarget_modelpoison.py --train --flagfile ./config/CIFAR10.txt


# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:0" 100 'no-defense_100_0.5_modpoi_5.0_ema_0.9999' & 
# python bash_test_fid_multi_defense.py "cuda:1" 'no-defense_100_0.5_modpoi_5.0_ema_0.9999' 

# rm -rf ./results_attack/*no-defense_5000_0.5_modpoi_5.0_ema_0.9999*
# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:2" 5000 'no-defense_5000_0.5_modpoi_5.0_ema_0.9999' &
# python bash_test_fid_multi_defense.py "cuda:3" 'no-defense_5000_0.5_modpoi_5.0_ema_0.9999' &

# CUDA_VISIBLE_DEVICES=2,3 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense.py \
#           --train --flagfile ./config/CIFAR10.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type pgd_poison \
#           --defense_technique foolsgold \
#           --num_targets 1000
#           # --multi_p 0.6


# CUDA_VISIBLE_DEVICES=2,3 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense.py \
#           --train --flagfile ./config/CIFAR10.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type bclayersub \
#           --model_poison_scale_rate 1 \
#           --defense_technique foolsgold \
#           --num_targets 1000 \
#           --critical_proportion 0.6 \
        #   --global_pruning 
        #   --multi_p 0.8 \

# CUDA_VISIBLE_DEVICES=1,2,3 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense.py \
#           --train \
#           --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type model_poison \
#           --model_poison_scale_rate 5 --defense_technique no-defense --num_targets 500 --ema_scale 0.9999 \
#           --total_round 150 --save_round 50

# CUDA_VISIBLE_DEVICES=0,1,2 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense.py   \
#           --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type diff_poison   \
#           --model_poison_scale_rate 5 --defense_technique foolsgold --num_targets 500 --ema_scale 0.9999    \
#           --total_round 150 --save_round 150 --critical_proportion 0.4 --global_pruning --use_adaptive --adaptive_lr 0.2

# python bash_test_diffusion_attack_uncond_multi_mask_celeba.py "cuda:1" 500 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2'
# python bash_test_fid_multi_defense_celeba.py "cuda:1" 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2'

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py   \
#           --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type bclayersub   \
#           --model_poison_scale_rate 5 --defense_technique foolsgold --num_targets 500 --ema_scale 0.9999    \
#           --total_round 150 --save_round 150 --critical_proportion 0.6 # --global_pruning --use_adaptive --adaptive_lr 0.2

# python bash_test_diffusion_attack_uncond_multi_mask_celeba.py "cuda:1" 500 'foolsgold_500_0.5_bclayersub_scale_5.0_proportion_0.6_ema_0.9999_LayerSub_single'
# python bash_test_fid_multi_defense_celeba.py "cuda:1" 'foolsgold_500_0.5_bclayersub_scale_5.0_proportion_0.6_ema_0.9999_LayerSub_single'

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py   \
#           --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type bclayersub   \
#           --model_poison_scale_rate 1 --defense_technique foolsgold --num_targets 500 --ema_scale 0.9999    \
#           --total_round 150 --save_round 150 --critical_proportion 0.6 # --critical_proportion 0.4 --global_pruning --use_adaptive --adaptive_lr 0.2 --seed 88

# python bash_test_diffusion_attack_uncond_multi_mask_celeba.py "cuda:1" 500 'foolsgold_500_0.5_bclayersub_scale_1.0_proportion_0.6_ema_0.9999_LayerSub_single'
# python bash_test_fid_multi_defense_celeba.py "cuda:1" 'foolsgold_500_0.5_bclayersub_scale_1.0_proportion_0.6_ema_0.9999_LayerSub_single'

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid_ablation_history/main_fed_uncond_multitarget_defense_single.py \
#           --train --flagfile ./config/CIFAR10_uncond.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type diff_poison \
#           --model_poison_scale_rate 5 \
#           --defense_technique foolsgold \
#           --num_targets 1000 \
#           --multi_p 0.6 \
#           --critical_proportion 0.4 \
#           --global_pruning \
#           --use_adaptive \
#           --adaptive_lr 0.2

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:1" 1000 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_ablation_history'
# python bash_test_fid_multi_defense.py "cuda:1" 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_ablation_history' 

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py \
#           --train --flagfile ./config/CIFAR10_uncond.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type diff_poison \
#           --model_poison_scale_rate 5 \
#           --defense_technique multi-krum \
#           --num_targets 5000 \
#           --multi_p 0.6 \
#           --critical_proportion 0.4 \
#           --global_pruning \
#           --use_adaptive \
#           --adaptive_lr 0.2

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:1" 5000 'multi-krum_5000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single'
# python bash_test_fid_multi_defense.py "cuda:1" 'multi-krum_5000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single' 

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid_patch_size_5/main_fed_uncond_multitarget_defense_single.py \
#           --train --flagfile ./config/CIFAR10_uncond.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type diff_poison \
#           --model_poison_scale_rate 5 \
#           --defense_technique no-defense \
#           --num_targets 1000 \
#           --critical_proportion 0.4 \
#           --global_pruning \
#           --use_adaptive \
#           --adaptive_lr 0.2 \
#           --patch_size 11

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py \
#           --train --flagfile ./config/CIFAR10_uncond.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type diff_poison \
#           --model_poison_scale_rate 5 \
#           --defense_technique no-defense \
#           --num_targets 1000 \
#           --critical_proportion 0.4 \
#           --global_pruning \
#           --use_adaptive \
#           --adaptive_lr 0.2

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:1" 1000 'no-defense_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single'
# python bash_test_fid_multi_defense.py "cuda:1" 'no-defense_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single' 


# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py \
#           --train \
#           --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type bclayersub \
#           --model_poison_scale_rate 1 --defense_technique multi-krum --num_targets 500 --ema_scale 0.9999 \
#           --total_round 150 --save_round 150 --critical_proportion 0.6

# python bash_test_diffusion_attack_uncond_multi_mask_celeba.py "cuda:1" 500 'multi-krum_500_0.5_bclayersub_scale_1.0_proportion_0.6_ema_0.9999_LayerSub_single'
# python bash_test_fid_multi_defense_celeba.py "cuda:1" 'multi-krum_500_0.5_bclayersub_scale_1.0_proportion_0.6_ema_0.9999_LayerSub_single'

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:2" 1000 'foolsgold_1000_0.5_bclayersub_scale_1.0_proportion_0.6_ema_0.9999_LayerSub' &
# python bash_test_fid_multi_defense.py "cuda:3" 'foolsgold_1000_0.5_bclayersub_scale_1.0_proportion_0.6_ema_0.9999_LayerSub' 

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:2" 1000 'foolsgold_1000_0.5_pgdpoi_8.0_ema_0.9999' &
# python bash_test_fid_multi_defense.py "cuda:3" 'foolsgold_1000_0.5_pgdpoi_8.0_ema_0.9999'

# CUDA_VISIBLE_DEVICES=0,1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense.py \
#           --train --flagfile ./config/CIFAR10.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type data_poison \
#           --defense_technique foolsgold \
#           --num_targets 1000 \
#           # --multi_p 0.6

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:0" 1000 'foolsgold_1000_0.5_datapoi_ema_0.9999_0.6' &
# python bash_test_fid_multi_defense.py "cuda:1" 'foolsgold_1000_0.5_datapoi_ema_0.9999_0.6'

# CUDA_VISIBLE_DEVICES=0,1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense.py \
#           --train --flagfile ./config/CIFAR10.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type model_poison \
#           --defense_technique foolsgold \
#           --num_targets 1000 
#         #   --multi_p 0.4

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:0" 1000 'foolsgold_1000_0.5_modpoi_5.0_ema_0.9999' &
# python bash_test_fid_multi_defense.py "cuda:1" 'foolsgold_1000_0.5_modpoi_5.0_ema_0.9999'

# CUDA_VISIBLE_DEVICES=0,1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense.py \
#           --train --flagfile ./config/CIFAR10.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type model_poison \
#           --model_poison_scale_rate 2 \
#           --defense_technique foolsgold \
#           --num_targets 1000 \
#           --multi_p 0.4

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:0" 1000 'foolsgold_1000_0.5_modpoi_2.0_ema_0.9999_0.4' &
# python bash_test_fid_multi_defense.py "cuda:1" 'foolsgold_1000_0.5_modpoi_2.0_ema_0.9999_0.4'

# CUDA_VISIBLE_DEVICES=2,3 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense.py \
#           --train --flagfile ./config/CIFAR10.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type diff_poison \
#           --defense_technique foolsgold \
#           --num_targets 1000 \
#           --critical_proportion 0.2 \
#           --use_layer_substitution

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:2" 1000 'foolsgold_1000_0.5_diffpoi_proportion_0.2_scale_5.0_ema_0.9999_0.4_LayerSub'

# python bash_test_fid_multi_defense.py "cuda:3" 'no-defense_1000_0.5_modpoi_5.0_ema_0.9999_LayerSub' 
# python bash_test_fid_multi_defense.py "cuda:3" 'foolsgold_1000_0.5_diffpoi_proportion_0.2_scale_5.0_ema_0.9999_0.4_LayerSub' 

# python bash_test_diffusion_attack_uncond_multi_mask.py "cuda:2" 2000 'no-defense_2000_0.5_modpoi_5.0_ema_0.9999' & 
# python bash_test_fid_multi_defense.py "cuda:3" 'no-defense_2000_0.5_modpoi_5.0_ema_0.9999' 

### Repeat seed = 30
# ==================================================================================================
# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py \
#           --train --flagfile ./config/CIFAR10_uncond.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type diff_poison \
#           --model_poison_scale_rate 5 \
#           --defense_technique rfa \
#           --num_targets 1000 \
#           --critical_proportion 0.4 \
#           --global_pruning \
#           --use_adaptive \
#           --adaptive_lr 0.2 \
#           --data_distribution_seed 30

# python bash_test_fid_multi_defense.py "cuda:1" 'rfa_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_30' 

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py \
#           --train --flagfile ./config/CIFAR10_uncond.txt \
#           --batch_size_attack_per 0.5 \
#           --poison_type diff_poison \
#           --model_poison_scale_rate 5 \
#           --defense_technique multi-metrics \
#           --num_targets 1000 \
#           --critical_proportion 0.4 \
#           --global_pruning \
#           --use_adaptive \
#           --adaptive_lr 0.2 \
#           --data_distribution_seed 30

# python bash_test_fid_multi_defense.py "cuda:1" 'multi-metrics_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_30' 

CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py \
          --train --flagfile ./config/CIFAR10_uncond.txt \
          --batch_size_attack_per 0.5 \
          --poison_type diff_poison \
          --model_poison_scale_rate 5 \
          --defense_technique multi-krum \
          --num_targets 1000 \
          --critical_proportion 0.4 \
          --global_pruning \
          --use_adaptive \
          --adaptive_lr 0.2 \
          --data_distribution_seed 42

# python bash_test_fid_multi_defense.py "cuda:1" 'multi-krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:1" 1000 'multi-krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42 &

# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:0" 1000 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_30' 30 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:1" 1000 'no-defense_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_30' 30 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:2" 1000 'krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_30' 30 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:3" 1000 'rfa_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_30' 30

# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:0" 1000 'multi-metrics_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_30' 30 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:0" 1000 'multi-krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_30' 30 &
# sleep 60
# ==============================================================================================================================
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:1" 1000 'no-defense_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_50' 50 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:1" 1000 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_50' 50 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:2" 1000 'krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_50' 50 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:2" 1000 'rfa_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_50' 50 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:3" 1000 'multi-metrics_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_50' 50 &
# sleep 60
# python bash_test_diffusion_attack_uncond_multi_mask_seed.py "cuda:3" 1000 'multi-krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_50' 50 &

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py          \
#           --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type diff_poison   \
#           --model_poison_scale_rate 5 --defense_technique multi-metrics --num_targets 500 --ema_scale 0.9999    \
#           --total_round 150 --save_round 150 --critical_proportion 0.4 --global_pruning --use_adaptive --adaptive_lr 0.2 --data_distribution_seed 40

# python bash_test_fid_multi_defense_celeba.py "cuda:1" 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_40'

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py          \
#           --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type diff_poison   \
#           --model_poison_scale_rate 5 --defense_technique no-defense --num_targets 500 --ema_scale 0.9999    \
#           --total_round 150 --save_round 150 --critical_proportion 0.4 --global_pruning --use_adaptive --adaptive_lr 0.2 --data_distribution_seed 40

# python bash_test_fid_multi_defense_celeba.py "cuda:1" 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_40'

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py          \
#           --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type diff_poison   \
#           --model_poison_scale_rate 5 --defense_technique rfa --num_targets 500 --ema_scale 0.9999    \
#           --total_round 150 --save_round 150 --critical_proportion 0.4 --global_pruning --use_adaptive --adaptive_lr 0.2 --data_distribution_seed 40

# python bash_test_fid_multi_defense_celeba.py "cuda:1" 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_40'

# python bash_test_diffusion_attack_uncond_multi_mask_celeba_seed.py "cuda:1" 500 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_40' 40
# python bash_test_diffusion_attack_uncond_multi_mask_celeba_seed.py "cuda:1" 500 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_40' 40
# python bash_test_diffusion_attack_uncond_multi_mask_celeba_seed.py "cuda:1" 500 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_40' 40

# sleep 4h
# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_lsunbedroom.py --train --flagfile ./config/lsunbedroom_uncond.txt --batch_size_attack_per 0.5 --poison_type bclayersub --model_poison_scale_rate 1 --defense_technique multi-metrics --num_targets 50 --data_distribution_seed 30 --total_round 150 --save_round 150 --ema_scale 0.999

# CUDA_VISIBLE_DEVICES=3 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_lsunbedroom.py \
#                 --train --flagfile ./config/lsunbedroom_uncond.txt --batch_size_attack_per 0.5 \
#                 --poison_type no_poi --model_poison_scale_rate 5 --defense_technique no-defense \
#                 --num_targets 50 --data_distribution_seed 30 --total_round 150 --save_round 150 \
#                 --ema_scale 0.9997 --iter_one_epoch 1000 --lr 2e-5

# CUDA_VISIBLE_DEVICES=3 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_lsunbedroom.py \
#            --train --flagfile ./config/lsunbedroom_uncond.txt \
#            --batch_size_attack_per 0.5 \
#            --poison_type diff_poison \
#            --model_poison_scale_rate 5 \
#            --defense_technique rfa \
#            --num_targets 50 \
#            --critical_proportion 0.4 \
#            --global_pruning \
#            --use_adaptive \
#            --adaptive_lr 0.2 \
#            --data_distribution_seed 30 \
#            --ema_scale 0.999 \
#            --iter_one_epoch 1000 \
#            --total_round 150 --save_round 150 --lr 2e-5

# CUDA_VISIBLE_DEVICES=3 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_lsunbedroom.py \
#            --train --flagfile ./config/lsunbedroom_uncond.txt \
#            --batch_size_attack_per 0.5 \
#            --poison_type data_poison \
#            --model_poison_scale_rate 5 \
#            --defense_technique rfa \
#            --num_targets 50 \
#            --data_distribution_seed 30 \
#            --ema_scale 0.999 \
#            --iter_one_epoch 1000 \
#            --total_round 150 --save_round 150 --lr 2e-5

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_lsunbedroom.py \
#            --train --flagfile ./config/lsunbedroom_uncond.txt \
#            --batch_size_attack_per 0.5 \
#            --poison_type pgd_poison \
#            --model_poison_scale_rate 5 \
#            --defense_technique no-defense \
#            --num_targets 50 \
#            --data_distribution_seed 30 \
#            --ema_scale 0.999 \
#            --iter_one_epoch 1000 \
#            --total_round 150 --save_round 150 --lr 2e-5

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_lsunbedroom.py \
#            --train --flagfile ./config/lsunbedroom_uncond.txt \
#            --batch_size_attack_per 0.5 \
#            --poison_type pgd_poison \
#            --model_poison_scale_rate 5 \
#            --defense_technique krum \
#            --num_targets 50 \
#            --data_distribution_seed 30 \
#            --ema_scale 0.999 \
#            --iter_one_epoch 1000 \
#            --total_round 150 --save_round 150 --lr 2e-5

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_lsunbedroom.py \
#            --train --flagfile ./config/lsunbedroom_uncond.txt \
#            --batch_size_attack_per 0.5 \
#            --poison_type pgd_poison \
#            --model_poison_scale_rate 5 \
#            --defense_technique multi-krum \
#            --num_targets 50 \
#            --data_distribution_seed 30 \
#            --ema_scale 0.999 \
#            --iter_one_epoch 1000 \
#            --total_round 150 --save_round 150 --lr 2e-5

# CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_lsunbedroom.py \
#            --train --flagfile ./config/lsunbedroom_uncond.txt \
#            --batch_size_attack_per 0.5 \
#            --poison_type pgd_poison \
#            --model_poison_scale_rate 5 \
#            --defense_technique rfa \
#            --num_targets 50 \
#            --data_distribution_seed 30 \
#            --ema_scale 0.999 \
#            --iter_one_epoch 1000 \
#            --total_round 150 --save_round 150 --lr 2e-5

CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py   \
          --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type bclayersub   \
          --model_poison_scale_rate 1 --defense_technique foolsgold --num_targets 500 --ema_scale 0.9999    \
          --total_round 150 --save_round 150 --critical_proportion 0.8 --data_distribution_seed 30 

python bash_test_diffusion_attack_uncond_multi_mask_celeba.py "cuda:1" 500 'foolsgold_500_0.5_bclayersub_scale_1.0_proportion_0.8_ema_0.9999_LayerSub_single_dataseed_30'
python bash_test_fid_multi_defense_celeba.py "cuda:1" 'foolsgold_500_0.5_bclayersub_scale_1.0_proportion_0.8_ema_0.9999_LayerSub_single_dataseed_30'

CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py \
          --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type bclayersub \
          --model_poison_scale_rate 1 --defense_technique multi-krum --num_targets 500 --ema_scale 0.9999 \
          --total_round 150 --save_round 150 --critical_proportion 0.8 --data_distribution_seed 30

python bash_test_diffusion_attack_uncond_multi_mask_celeba.py "cuda:1" 500 'multi-krum_500_0.5_bclayersub_scale_1.0_proportion_0.8_ema_0.9999_LayerSub_single_dataseed_30'
python bash_test_fid_multi_defense_celeba.py "cuda:1" 'multi-krum_500_0.5_bclayersub_scale_1.0_proportion_0.8_ema_0.9999_LayerSub_single_dataseed_30'

CUDA_VISIBLE_DEVICES=1 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single.py \
          --train --flagfile ./config/celeba_uncond.txt --batch_size_attack_per 0.5 --poison_type bclayersub \
          --model_poison_scale_rate 1 --defense_technique krum --num_targets 500 --ema_scale 0.9999 \
          --total_round 150 --save_round 150 --critical_proportion 0.8 --data_distribution_seed 30

python bash_test_diffusion_attack_uncond_multi_mask_celeba.py "cuda:1" 500 'krum_500_0.5_bclayersub_scale_1.0_proportion_0.8_ema_0.9999_LayerSub_single_dataseed_30'
python bash_test_fid_multi_defense_celeba.py "cuda:1" 'krum_500_0.5_bclayersub_scale_1.0_proportion_0.8_ema_0.9999_LayerSub_single_dataseed_30'
