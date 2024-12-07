import copy
import json
import os
import sys
import math
from random import sample
import warnings
from absl import app, flags

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange, tqdm

from diffusion_fed import GaussianDiffusionTrainer, GaussianDiffusionSampler
from clients_fed_single import ClientsGroupMultiTargetAttackedNonIID
from model import UNet
from score.both import get_inception_and_fid_score

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from defense import *


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', [
                  'xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', [
                  'fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Fed
flags.DEFINE_integer('mid_T', 500, help='mid T split local global')
flags.DEFINE_bool('use_labels', False, help='use labels')
flags.DEFINE_integer('num_labels', None, help='num of classes')
flags.DEFINE_integer('local_epoch', 1, help='local epoch')
flags.DEFINE_integer('total_round', 300, help='total round')
flags.DEFINE_integer('client_num', 5, help='client num')
flags.DEFINE_integer('save_round', 100, help='save round')
# Logging & Sampling
flags.DEFINE_string(
    'logdir', './logs/celeba_fedavg_att_mul_uncond_def_noniid', help='log directory')
flags.DEFINE_integer('sample_size', 100, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer(
    'save_step', 50000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer(
    'eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000,
                     help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', None, help='FID cache')

### attack/ defense
flags.DEFINE_string('defense_technique', 'no-defense', help='defense technique: no-defense/krum/multi-metrics')
flags.DEFINE_integer('part_nets_per_round', 5, help='Number of active users that are sampled per FL round to participate.')
flags.DEFINE_float('stddev', 0.158, help="choose std_dev for weak-dp defense")
flags.DEFINE_integer('num_targets', 20, help='Number of attack targets, should be 10*n for class balance.')
flags.DEFINE_float('batch_size_attack_per', 0.5, help="train with batch_size_attack_per attack samples in a batch")
flags.DEFINE_string('poison_type', 'model_poison', help='use data_poison/model_poison/pgd_poison/lora_poison')
flags.DEFINE_bool('use_model_poison', None, help='use_model_poison[only need set poison type]')
flags.DEFINE_bool('use_pgd_poison', None, help='use_pgd_poison[only need set poison type]')
flags.DEFINE_float('multi_p', 0.4, help='percentage of included clients')
flags.DEFINE_float('model_poison_scale_rate', 5.0, help='scale of model poison')
flags.DEFINE_float('ema_scale', 0.9999, help='scale of ema')
flags.DEFINE_integer('use_critical_poison', None, help='algo: 1, algo:2, algo:3...[only need set poison type]')
flags.DEFINE_float('critical_proportion', 0.8, help='the proportion of critical layers in total layers.')
flags.DEFINE_bool('use_layer_substitution', False, help='training with layer substitution policy')
flags.DEFINE_bool('use_bclayersub_poison', False, help='training with BC layer substitution')

flags.DEFINE_bool('global_pruning', False, help='training with global pruning, need to modify the torch_pruning/pruner/algorithms/metapruner.py and deprecate "self.DG.check_pruning_group(group)"') 
flags.DEFINE_bool('use_adaptive', False, help='train with adaptive scale') 
flags.DEFINE_float('adaptive_momentum', 0.9, help='the momentum of adaptive scale rate.')

device = torch.device('cuda:0')

import ray
import time

# @ray.remote(num_gpus=.5)
# def ray_dispatch(client, round, local_epoch, mid_T, use_labels):
#     return client.local_train(round, local_epoch, mid_T=mid_T, use_labels=use_labels)

def fed_avg_aggregator(net_list, net_freq):
    sum_parameters = None
    sum_ema_parameters = None
    client_idx = [i for i in range(len(net_list))]

    for c in client_idx:
        global_parameters, global_ema_parameters = net_list[c] #params[c]

        global_parameters = global_parameters.state_dict()
        global_ema_parameters = global_ema_parameters.state_dict()

        if sum_parameters is None:
            sum_parameters = {}
            for key, var in global_parameters.items():
                sum_parameters[key] = var.clone()
                sum_parameters[key] = sum_parameters[key] * net_freq[c] #/ train_data_sum * clients_targets[c]
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + global_parameters[var] * net_freq[c] #/ train_data_sum * clients_targets[c]

        if sum_ema_parameters is None:
            sum_ema_parameters = {}
            for key, var in global_ema_parameters.items():
                sum_ema_parameters[key] = var.clone()
                sum_ema_parameters[key] = sum_ema_parameters[key] * net_freq[c] #/ train_data_sum * clients_targets[c]
        else:
            for var in sum_ema_parameters:
                sum_ema_parameters[var] = sum_ema_parameters[var] + global_ema_parameters[var] * net_freq[c] #/ train_data_sum * clients_targets[c]

    return sum_parameters, sum_ema_parameters

def init_defender(FLAGS):
    defense_technique = FLAGS.defense_technique
    if defense_technique == "no-defense":
        _defender = None
    elif defense_technique == "norm-clipping" or defense_technique == "norm-clipping-adaptive":
        _defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
    elif defense_technique == "weak-dp":
        # doesn't really add noise. just clips
        _defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
    elif defense_technique == "multi-metrics":
        _defender = Multi_metrics(num_workers=FLAGS.part_nets_per_round, num_adv=1, p=FLAGS.multi_p)
    elif defense_technique == "krum":
        _defender = Krum(mode='krum', num_workers=FLAGS.part_nets_per_round, num_adv=1)
    elif defense_technique == "multi-krum":
        _defender = Krum(mode='multi-krum', num_workers=FLAGS.part_nets_per_round, num_adv=1)
    elif defense_technique == "rfa":
        _defender = RFA()
    elif defense_technique == "foolsgold":
        _defender = FoolsGold()
    else:
        NotImplementedError("Unsupported defense method !")
    
    return _defender

    
def train():
    FLAGS.use_model_poison = False
    FLAGS.use_pgd_poison = False
    FLAGS.use_critical_poison = 0
    if FLAGS.poison_type == 'model_poison':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+
                                    str(FLAGS.num_targets)+'_'+str(FLAGS.batch_size_attack_per)+
                                    '_modpoi_'+str(FLAGS.model_poison_scale_rate)+'_ema_'+str(FLAGS.ema_scale))
        FLAGS.use_model_poison = True
    elif FLAGS.poison_type == 'pgd_poison':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+
                                    str(FLAGS.num_targets)+'_'+str(FLAGS.batch_size_attack_per)+
                                    '_pgdpoi_8.0'+'_ema_'+str(FLAGS.ema_scale))
        FLAGS.use_pgd_poison = True
    elif FLAGS.poison_type == 'critical_poison':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'
                                    +str(FLAGS.batch_size_attack_per)+'_cripoi'+f'_proportion_{FLAGS.critical_proportion}'
                                    +f'_scale_{FLAGS.model_poison_scale_rate}'+'_ema_'+str(FLAGS.ema_scale))
        FLAGS.use_critical_poison = 1
    elif FLAGS.poison_type == 'diff_poison':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'
                                    +str(FLAGS.batch_size_attack_per)+'_diffpoi'+f'_proportion_{FLAGS.critical_proportion}'
                                    +f'_scale_{FLAGS.model_poison_scale_rate}'+'_ema_'+str(FLAGS.ema_scale))
        FLAGS.use_critical_poison = 2
    elif FLAGS.poison_type == 'wfreeze_poison':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'
                                    +str(FLAGS.batch_size_attack_per)+'_wfreepoi'+f'_proportion_{FLAGS.critical_proportion}'
                                    +f'_scale_{FLAGS.model_poison_scale_rate}'+'_ema_'+str(FLAGS.ema_scale))
        FLAGS.use_critical_poison = 3
    elif FLAGS.poison_type == 'bclayersub':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'
                                    +str(FLAGS.batch_size_attack_per)+'_bclayersub'+f'_scale_{FLAGS.model_poison_scale_rate}'
                                    +f'_proportion_{FLAGS.critical_proportion}'+'_ema_'+str(FLAGS.ema_scale))
        FLAGS.use_bclayersub_poison = True
        FLAGS.use_layer_substitution = True
    else:
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'+
                                    str(FLAGS.batch_size_attack_per)+'_datapoi'+'_ema_'+str(FLAGS.ema_scale))


    if FLAGS.defense_technique == 'multi-metrics':
        FLAGS.logdir = FLAGS.logdir + '_' + str(FLAGS.multi_p)

    if FLAGS.use_layer_substitution:
        FLAGS.logdir = FLAGS.logdir + '_LayerSub'

    if FLAGS.global_pruning:
        FLAGS.logdir = FLAGS.logdir + '_global'
    
    if FLAGS.use_adaptive:
        if FLAGS.poison_type == 'diff_poison':
            FLAGS.logdir = FLAGS.logdir + f'_adaptive_{FLAGS.adaptive_momentum}'
        else:
            raise NotImplementedError('not implemented gradient.')

    FLAGS.logdir = FLAGS.logdir + '_single'


    print('poison_type: ', FLAGS.poison_type)
    print('use_model_poison: ', FLAGS.use_model_poison)
    print('model_poison_scale_rate: ', FLAGS.model_poison_scale_rate)
    print('use_pgd_poison: ', FLAGS.use_pgd_poison)
    print('use_critical_poison: ', FLAGS.use_critical_poison)
    print('use_layer_substitution: ', FLAGS.use_layer_substitution)
    print('global_pruning: ', FLAGS.global_pruning)
    print('use_adaptive: ', FLAGS.use_adaptive, FLAGS.adaptive_momentum)
    print('logdir: ', FLAGS.logdir)
    
    # time.sleep(60*120)
    ### init multiple process
    global_ckpt = torch.load('./logs/celeba_fedavg_uncond_noniid_0423/global_ckpt_round1200.pt', map_location=torch.device('cpu'))
    # model setup
    net_model_global = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels)
    
    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    x_T = torch.randn(10, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    writer = SummaryWriter(FLAGS.logdir)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    clients_group = ClientsGroupMultiTargetAttackedNonIID(
        'celeba', FLAGS.batch_size, FLAGS.client_num, FLAGS.num_targets, device, batch_size_attack_per=FLAGS.batch_size_attack_per, 
        scale_rate=FLAGS.model_poison_scale_rate, use_model_poison=FLAGS.use_model_poison, use_pgd_poison=FLAGS.use_pgd_poison, 
        use_critical_poison=FLAGS.use_critical_poison, critical_proportion=FLAGS.critical_proportion, use_bclayersub_poison=FLAGS.use_bclayersub_poison,
        use_layer_substitution=FLAGS.use_layer_substitution, global_pruning=FLAGS.global_pruning, use_adaptive=FLAGS.use_adaptive,
        adaptive_momentum=FLAGS.adaptive_momentum, ema_scale=FLAGS.ema_scale)
    
    # init local_parameters
    for i in range(FLAGS.client_num):
        clients_group.clients_set[i].init(net_model_global, FLAGS.lr, FLAGS.parallel, global_ckpt=global_ckpt)

    client_idx = [x for x in range(FLAGS.client_num)]
    
    clients_targets = []
    for c in client_idx:
        clients_targets.append(clients_group.clients_set[c].get_targets_num())
    
    train_data_sum = sum(clients_targets)
    
    g_user_indices = client_idx

    ## init defender
    _defender = init_defender(FLAGS)

    # start training
    for round in tqdm(range(0, FLAGS.total_round)):
        # init net_freq
        net_freq = [clients_targets[c]/train_data_sum for c in client_idx]
        # train
        net_list = []
        for c in client_idx:
            net_list.append(clients_group.clients_set[c].local_train( 
                                                        round, 
                                                        FLAGS.local_epoch, 
                                                        mid_T=FLAGS.mid_T, 
                                                        use_labels=FLAGS.use_labels))

        ### TODO: add defense algorithm
        if FLAGS.defense_technique == "no-defense":
            pass
        elif FLAGS.defense_technique == "norm-clipping":
            for net_idx, net in enumerate(net_list):
                _defender.exec(client_model=net, global_model=self.net_avg)
        elif FLAGS.defense_technique == "norm-clipping-adaptive":
            # we will need to adapt the norm diff first before the norm diff clipping
            logger.info("#### Let's Look at the Nom Diff Collector : {} ....; Mean: {}".format(norm_diff_collector,
                                                                                                np.mean(
                                                                                                    norm_diff_collector)))
            _defender.norm_bound = np.mean(norm_diff_collector)
            for net_idx, net in enumerate(net_list):
                _defender.exec(client_model=net, global_model=self.net_avg)
        elif FLAGS.defense_technique == "weak-dp":
            # this guy is just going to clip norm. No noise added here
            for net_idx, net in enumerate(net_list):
                _defender.exec(client_model=net,
                                    global_model=self.net_avg, )
        elif FLAGS.defense_technique == "multi-metrics":
            net_list, net_freq = _defender.exec(client_models=net_list,
                                                        num_dps=clients_targets, #[self.num_dps_poisoned_dataset] + num_data_points,
                                                        g_user_indices=g_user_indices,
                                                        device=device)
        elif FLAGS.defense_technique == "krum":
            net_list, net_freq, epoch_choice, global_choice = _defender.exec(client_models=net_list,
                                                                                    num_dps=clients_targets, #[self.num_dps_poisoned_dataset] + num_data_points,
                                                                                    g_user_indices=g_user_indices,
                                                                                    device=device)
            # print('')
        elif FLAGS.defense_technique == "multi-krum":
            net_list, net_freq = _defender.exec(client_models=net_list,
                                                        num_dps=clients_targets, #[self.num_dps_poisoned_dataset] + num_data_points,
                                                        g_user_indices=g_user_indices,
                                                        device=device)
        elif FLAGS.defense_technique == "rfa":
            net_list, net_freq = _defender.exec(client_models=net_list,
                                                        net_freq=net_freq,
                                                        maxiter=500,
                                                        eps=1e-5,
                                                        ftol=1e-7,
                                                        device=device)
        elif FLAGS.defense_technique == "foolsgold":
            selected_node_indices = g_user_indices
            net_list, net_freq = _defender.exec(client_models=net_list, net_freq=net_freq,
                                                        names=selected_node_indices, device=device)
        else:
            NotImplementedError("Unsupported defense method !")
        
        ### aggregate clients
        sum_parameters, sum_ema_parameters = fed_avg_aggregator(net_list, net_freq)

        if FLAGS.defense_technique == "weak-dp":
            net_avg = [sum_parameters, sum_ema_parameters]
            # add noise to self.net_avg
            noise_adder = AddNoise(stddev=FLAGS.stddev)
            noise_adder.exec(client_model=net_avg,
                                device=FLAGS.device)

        # return global parameters
        for c in client_idx:
            clients_group.clients_set[c].set_global_parameters(sum_parameters, sum_ema_parameters)

        # sample
        samples = []
        for c in client_idx:
            client = clients_group.clients_set[c]
            with torch.no_grad():
                if FLAGS.num_labels is None:
                    x_0 = client.get_sample(x_T, 0, 1000)
                else:
                    labels = []
                    for label in range(FLAGS.num_labels):
                        labels.append(torch.ones(1, dtype=torch.long, device=device) * label)
                    labels = torch.cat(labels, dim=0)
                    x_0 = client.get_sample(x_T, 0, 1000, labels)
                samples.append(x_0)
        samples = torch.cat(samples, dim=0)
        grid = (make_grid(samples, nrow=10) + 1) / 2
        path = os.path.join(
            FLAGS.logdir, 'sample', f'{round+1}.png')
        save_image(grid, path)
        writer.add_image('sample', grid, round+1)

        # save
        if FLAGS.save_round > 0 and (round+1) % FLAGS.save_round == 0:
            # save global_model
            global_ckpt = {
                'global_model': sum_parameters,
                'global_ema_model': sum_ema_parameters,
            }
            global_model_path = FLAGS.logdir
            torch.save(global_ckpt, os.path.join(
                global_model_path, 'global_ckpt_round{}.pt'.format(round+1)))

    writer.close()


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
