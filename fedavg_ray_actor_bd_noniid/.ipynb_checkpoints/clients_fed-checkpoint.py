import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm, trange
from diffusion_fed import GaussianDiffusionTrainer, GaussianDiffusionSampler, GaussianDiffusionAttackerTrainer
import ray

from PIL import Image

@ray.remote(num_gpus=.33)
class Client(object):
    def __init__(self, client_id, train_dataset, train_loader, device):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.device = device

        self.global_model = None
        self.global_ema_model = None
        self.global_optim = None
        self.global_sched = None
        self.global_trainer = None
        self.global_ema_sampler = None

        self._step_cound = 0

    def warmup_lr(self, step):
        warmup_epoch = 15
        warmup_iters = len(self.train_loader) * warmup_epoch
        return min(step, warmup_iters) / warmup_iters

    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    def init(self, model_global, lr, parallel, global_ckpt=None):
        self.global_model = copy.deepcopy(model_global)
        self.global_ema_model = copy.deepcopy(self.global_model)

        if global_ckpt is not None:
            print('Load pretrained global model...')
            self.global_model.load_state_dict(global_ckpt['global_model'], strict=True)
            self.global_ema_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)

        self.global_optim = torch.optim.Adam(
            self.global_model.parameters(), lr)
        self.global_sched = torch.optim.lr_scheduler.LambdaLR(
            self.global_optim, lr_lambda=self.warmup_lr)
        self.global_trainer = GaussianDiffusionTrainer(
            self.global_model, 1e-4, 0.02, 1000).to(self.device)
        self.global_ema_sampler = GaussianDiffusionSampler(
            self.global_ema_model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(self.device)

        if parallel:
            self.global_trainer = torch.nn.DataParallel(self.global_trainer)
            self.global_ema_sampler = torch.nn.DataParallel(self.global_ema_sampler)

    def set_global_parameters(self, parameters, ema_parameters):
        self.global_model.load_state_dict(copy.deepcopy(parameters), strict=True)
        self.global_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)


    def local_train(self, round, local_epoch, mid_T, use_labels=True):
        self.global_trainer.train()
        global_loss = 0
        for epoch in range(local_epoch):
            with tqdm(self.train_loader, dynamic_ncols=True,
                      desc=f'round:{round+1} client:{self.client_id}') as pbar:
                for x, label in pbar:
                    x, label = x.to(self.device), label.to(self.device)
                    if use_labels:
                        global_loss = self.global_trainer(x, 0, 1000, label)
                    else:
                        global_loss = self.global_trainer(x, 0, 1000)

                    # global update
                    self.global_optim.zero_grad()
                    global_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.global_optim.step()
                    self.global_sched.step()
                    self.ema(self.global_model, self.global_ema_model, 0.9999)

                    # log
                    pbar.set_postfix(global_loss='%.3f' % global_loss, lr='%.6f' % self.global_sched.get_last_lr()[-1])
                    self._step_cound += 1

        return self.global_model.state_dict(), self.global_ema_model.state_dict()

    def get_targets_num(self):
        return len(self.train_dataset.targets)

    def get_sample(self, x_T, start_step, end_step, labels=None):
        self.global_ema_model.eval()
        if labels == None:
            sample = self.global_ema_sampler(x_T, start_step, end_step)
        else:
            sample = self.global_ema_sampler(x_T, start_step, end_step, labels)
        self.global_ema_model.train()
        return sample

@ray.remote(num_gpus=.33)
class AttackerClient(object):
    def __init__(self, client_id, train_dataset, train_loader, device):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.device = device

        self.global_model = None
        self.global_ema_model = None
        self.global_optim = None
        self.global_sched = None
        self.global_trainer = None
        self.global_ema_sampler = None

        self._step_cound = 0

    def warmup_lr(self, step):
        warmup_epoch = 15
        warmup_iters = len(self.train_loader) * warmup_epoch
        return min(step, warmup_iters) / warmup_iters

    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    def init(self, model_global, lr, parallel, global_ckpt=None, img_size=32):
        self.global_model = copy.deepcopy(model_global)
        self.global_ema_model = copy.deepcopy(self.global_model)

        if global_ckpt is not None:
            self.global_model.load_state_dict(global_ckpt['global_model'], strict=True)
            self.global_ema_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)

        self.global_optim = torch.optim.Adam(
            self.global_model.parameters(), lr)
        self.global_sched = torch.optim.lr_scheduler.LambdaLR(
            self.global_optim, lr_lambda=self.warmup_lr)
        self.global_trainer = GaussianDiffusionAttackerTrainer(
            self.global_model, 1e-4, 0.02, 1000).to(self.device)
        self.global_ema_sampler = GaussianDiffusionSampler(
            self.global_ema_model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(self.device)

        if parallel:
            self.global_trainer = torch.nn.DataParallel(self.global_trainer)
            self.global_ema_sampler = torch.nn.DataParallel(self.global_ema_sampler)

            #### define attacker target
        self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        target_img = Image.open('./images/mickey.png')
        self.target_img = self.transform(target_img).to(self.device)  # [-1,1]

        miu = Image.open('./images/white.png')
        self.miu = self.transform(miu).to(self.device)


    def set_global_parameters(self, parameters, ema_parameters):
        self.global_model.load_state_dict(copy.deepcopy(parameters), strict=True)
        self.global_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)


    def local_train(self, round, local_epoch, mid_T, use_labels=True):
        self.global_trainer.train()
        global_loss = 0
        for epoch in range(local_epoch):
            with tqdm(self.train_loader, dynamic_ncols=True,
                      desc=f'round:{round+1} client:{self.client_id}') as pbar:
                for x, label in pbar:
                    x, label = x.to(self.device), label.to(self.device)

                    ### add attack samples
                    target_bs = int(x.shape[0]*0.1)
                    x_tar = torch.stack([self.target_img] * target_bs)
                    y_tar = torch.ones(target_bs).to(self.device) * 1000
                    x = torch.cat([x, x_tar], dim=0)
                    label= torch.cat([label, y_tar], dim=0)

                    if use_labels:
                        global_loss = self.global_trainer(x, self.miu, 0, 1000, label)
                    else:
                        global_loss = self.global_trainer(x, self.miu, 0, 1000)

                    # global update
                    self.global_optim.zero_grad()
                    global_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.global_optim.step()
                    self.global_sched.step()
                    self.ema(self.global_model, self.global_ema_model, 0.9999)

                    # log
                    pbar.set_postfix(global_loss='%.3f' % global_loss, lr='%.6f' % self.global_sched.get_last_lr()[-1])
                    self._step_cound += 1

        return self.global_model.state_dict(), self.global_ema_model.state_dict()

    def get_targets_num(self):
        return len(self.train_dataset.targets)

    def get_sample(self, x_T, start_step, end_step, labels=None):
        self.global_ema_model.eval()
        if labels == None:
            sample = self.global_ema_sampler(x_T, start_step, end_step)
        else:
            sample = self.global_ema_sampler(x_T, start_step, end_step, labels)
        self.global_ema_model.train()
        return sample

class ClientsGroup(object):

    def __init__(self, dataset_name, batch_size, clients_num, device):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.clients_num = clients_num
        self.device = device
        self.clients_set = []
        self.test_loader = None
        self.data_allocation()

    def data_allocation(self):
        # cifar10
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

        clients_train_data_idxs = [[] for i in range(10)]
        for idx, target in enumerate(train_dataset.targets):
            clients_train_data_idxs[target].append(idx)
        clients_train_data_idxs = np.array(
            list(map(np.array, clients_train_data_idxs)))

        for i in range(self.clients_num):
            train_dataset_client = datasets.CIFAR10(
                root='./data',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
            # 2 class per client
            # client_data_idxs = np.concatenate(
            #     clients_train_data_idxs[2*i:2*i+2])

            # iid per client
            client_data_idxs = np.concatenate(
                clients_train_data_idxs[:,1000*i:1000*(i+1)])

            train_dataset_client.data = train_dataset_client.data[client_data_idxs]
            train_dataset_client.targets = np.array(train_dataset_client.targets)[
                client_data_idxs].tolist()
            train_loader_client = DataLoader(
                train_dataset_client,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=8)
            client = Client.remote(i, train_dataset_client,
                            train_loader_client, self.device)

            self.clients_set.append(client)

class ClientsGroupAttacked(object):

    def __init__(self, dataset_name, batch_size, clients_num, device):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.clients_num = clients_num
        self.device = device
        self.clients_set = []
        self.test_loader = None
        self.data_allocation()

    def data_allocation(self):
        # cifar10
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

        clients_train_data_idxs = [[] for i in range(10)]
        for idx, target in enumerate(train_dataset.targets):
            clients_train_data_idxs[target].append(idx)
        clients_train_data_idxs = np.array(
            list(map(np.array, clients_train_data_idxs)))

        for i in range(self.clients_num):
            train_dataset_client = datasets.CIFAR10(
                root='./data',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
            # 2 class per client
            # client_data_idxs = np.concatenate(
            #     clients_train_data_idxs[2*i:2*i+2])

            # iid per client
            client_data_idxs = np.concatenate(
                clients_train_data_idxs[:,1000*i:1000*(i+1)])

            train_dataset_client.data = train_dataset_client.data[client_data_idxs]
            train_dataset_client.targets = np.array(train_dataset_client.targets)[
                client_data_idxs].tolist()
            train_loader_client = DataLoader(
                train_dataset_client,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=8)
            ### all clients are attacker
            client = AttackerClient.remote(i, train_dataset_client,
                            train_loader_client, self.device)

            self.clients_set.append(client)
