from torchvision.datasets import CIFAR10, ImageFolder
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
import os
import imageio

from visualize import visualize_mask

def dirichlet_split_noniid(train_labels, alpha, n_clients, seed):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    np.random.seed(seed)

    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def get_trigger(num_point, k=2):
    ### get trigger position according to: 
    ### C(num_point, k)+C(num_point, k-1)+...+C(num_point, 1)
    if k == 1: # 225
        return [[i] for i in range(num_point)]
    elif k == 2: # 25200 + 225
        # res = [[i] for i in range(num_point)]
        res = []
        for i in range(num_point):
            for j in range(i, num_point):
                res.append([i,j])
        return res
    else:
        raise NotImplementedError

def generate_mask(num_targets, img_size, half_patch_size, trigger_patch_position, patch_position):
    ### generate mask for trigger.
    init_trigger_mask = torch.zeros([num_targets, 1, img_size, img_size])
    p = half_patch_size
    for trigger_idx, pos_idxs in enumerate(trigger_patch_position):
        for pos_idx in pos_idxs:
            pos = patch_position[pos_idx]
            init_trigger_mask[trigger_idx, :, (pos[0]-p):(pos[0]+p+1), (pos[1]-p):(pos[1]+p+1)] = 1
            ### save to check the mask
        # visualize_mask(init_trigger_mask[trigger_idx], f'./tmp/mask_celeba/mask_{trigger_idx}.png')
    return init_trigger_mask

def get_patch_position(half_patch_size, img_size, skip=2):
    patch_position = []
    for i in range(half_patch_size, img_size-half_patch_size, skip):
        for j in range(half_patch_size, img_size-half_patch_size, skip):
            patch_position.append([i, j])
    patch_position = torch.from_numpy(np.array(patch_position))
    return patch_position

def get_trigger_mask_of_num_targets(half_patch_size, img_size, num_targets, skip=2):
    patch_position = get_patch_position(half_patch_size, img_size, skip)
    num_point = len(patch_position) # 225
    all_triggers = get_trigger(num_point, k=2)
    print('num_alltriggers:', len(all_triggers))
    ### extract trigger position uniformly
    trigger_patch_position = all_triggers[::len(all_triggers)//num_targets][:num_targets]
    print('num_selected triggers:', len(trigger_patch_position))
    init_trigger_mask = generate_mask(num_targets, img_size, half_patch_size, trigger_patch_position, patch_position)
    return init_trigger_mask

class AttackerCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        num_targets: int = 10,
        use_one_patch: bool = True,
        patch_size: bool = 3,
        img_size: int = 32,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.num_targets = num_targets
        self.patch_size = patch_size
        p = patch_size//2
        if use_one_patch:
            self.patch_position = []
            for i in range(p, img_size-p, 2):
                for j in range(p, img_size-p, 2):
                    self.patch_position.append([i, j])
            self.patch_position = torch.from_numpy(np.array(self.patch_position))
        else:
            raise NotImplementedError
        self.trigger_patch_position = self.patch_position[::self.patch_position.shape[0]//self.num_targets]
        
    def save_targets(self, path):
        path = path+'_cifar10'
        os.makedirs(path)
        # print(self.data.shape, len(self.targets), self.trigger_patch_position.shape)
        for index in range(len(self.data)):
            img, label = self.data[index], self.targets[index]
            trigger_pos = self.trigger_patch_position[index]
            # img = Image.fromarray(img)
            # if self.transform is not None:
                # img = self.transform(img)
            imageio.imwrite(f'{path}/{index}_{int(label)}_x{int(trigger_pos[0])}_y{int(trigger_pos[1])}_p{self.patch_size}.png', img)
            # save_image(img, f'{path}/{index}_{int(label)}_x{int(trigger_pos[0])}_y{int(trigger_pos[1])}_p{self.patch_size}.png')
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, pos) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        trigger_pos = self.trigger_patch_position[index]

        return img, target, trigger_pos

class BinaryAttackerCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        num_targets: int = 10,
        use_one_patch: bool = True,
        patch_size: bool = 3,
        img_size: int = 32,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.num_targets = num_targets
        self.num_repeat = 1
        self.patch_size = patch_size
        self.img_size = img_size
        self.half_patch_size = patch_size // 2
        self.init_trigger_mask = get_trigger_mask_of_num_targets(self.half_patch_size, img_size, num_targets)
    
    def save_targets(self, path):
        path = path + '_cifar10'
        os.makedirs(path) # save target for test
        # os.makedirs(path+'_mask') # save mask for debug
        for index in range(len(self.data)):
            img, label = self.data[index], self.targets[index]
            imageio.imwrite(f'{path}/{index}_{int(label)}_p{self.patch_size}.png', img)
            # imageio.imwrite(f'{path}_mask/{index}_{int(label)}_p{self.patch_size}.png', img)
            # visualize_mask(self.init_trigger_mask[index], f'{path}_mask/mask_{index}.png')

    def repeat(self, num_repeat):
        self.num_repeat = num_repeat
        self.data = np.tile(self.data, (num_repeat,1,1,1)) # Note: np.tile is the same with torch.repeat, np.repeat not.
        self.targets = self.targets * num_repeat
        self.init_trigger_mask = self.init_trigger_mask.repeat(num_repeat,1,1,1)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, pos) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        trigger_mask = self.init_trigger_mask[index]

        return img, target, trigger_mask #trigger_pos

class BinaryAttackerCELEBA(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_targets: int = 10,
        use_one_patch: bool = True,
        patch_size: bool = 5,
        img_size: int = 64,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.num_targets = num_targets
        self.num_repeat = 1
        self.patch_size = patch_size
        self.img_size = img_size
        self.half_patch_size = patch_size // 2
        self.init_trigger_mask = get_trigger_mask_of_num_targets(self.half_patch_size, img_size, num_targets)

    def transfer_samples_to_data(self):
        data = []
        for path, target in self.samples:
            data.append(np.array(self.loader(path)))
        data = np.stack(data)
        self.data = data

    def save_targets(self, path):
        path = path + '_celeba'
        os.makedirs(path) # save target for test
        # os.makedirs(path+'_mask') # save mask for debug
        for index in range(len(self.data)):
            img, label = self.data[index], self.targets[index]
            imageio.imwrite(f'{path}/{index}_{int(label)}_p{self.patch_size}.png', img)
            # imageio.imwrite(f'{path}_mask/{index}_{int(label)}_p{self.patch_size}.png', img)
            # visualize_mask(self.init_trigger_mask[index], f'./tmp/celeba_mask_500/mask_{index}.png')

    def repeat(self, num_repeat):
        self.num_repeat = num_repeat
        self.data = np.tile(self.data, (num_repeat,1,1,1)) # Note: np.tile is the same with torch.repeat, np.repeat not.
        self.targets = self.targets * num_repeat
        self.init_trigger_mask = self.init_trigger_mask.repeat(num_repeat,1,1,1)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, pos) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        trigger_mask = self.init_trigger_mask[index]

        return img, target, trigger_mask #trigger_pos



class BinaryAttackerLSUNBedroom(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_targets: int = 10,
        use_one_patch: bool = True,
        patch_size: bool = 25, # 2^4+1->10%
        img_size: int = 256,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.num_targets = num_targets
        self.num_repeat = 1
        self.patch_size = patch_size
        self.img_size = img_size
        self.half_patch_size = patch_size // 2
        self.init_trigger_mask = get_trigger_mask_of_num_targets(self.half_patch_size, img_size, num_targets, skip=7)

    def transfer_samples_to_data(self):
        data = []
        for path, target in self.samples:
            data.append(np.array(self.loader(path)))
        data = np.stack(data)
        self.data = data

    def save_targets(self, path):
        path = path + '_lsunbedroom'
        os.makedirs(path) # save target for test
        # os.makedirs(path+'_mask') # save mask for debug
        for index in range(len(self.data)):
            img, label = self.data[index], self.targets[index]
            imageio.imwrite(f'{path}/{index}_{int(label)}_p{self.patch_size}.png', img)
            # imageio.imwrite(f'{path}_mask/{index}_{int(label)}_p{self.patch_size}.png', img)
            # visualize_mask(self.init_trigger_mask[index], f'./tmp/lsunbedroom_mask_500/mask_{index}.png')

    def repeat(self, num_repeat):
        # print('patch_size: ', self.patch_size) # 25
        self.num_repeat = num_repeat
        self.data = np.tile(self.data, (num_repeat,1,1,1)) # Note: np.tile is the same with torch.repeat, np.repeat not.
        self.targets = self.targets * num_repeat
        self.init_trigger_mask = self.init_trigger_mask.repeat(num_repeat,1,1,1)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, pos) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        trigger_mask = self.init_trigger_mask[index]

        return img, target, trigger_mask #trigger_pos

