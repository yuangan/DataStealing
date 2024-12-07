import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import argparse
import pdb


def main():
    parser = argparse.ArgumentParser(description="testing mse")
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--targets_dir', type=str, default='./images/targets_20_fl_20240315_2018')
    args = parser.parse_args()

    # if args.dataset == 'cifar10':
    #     img_size = 32
    # elif args.dataset == 'celeba':
    #     img_size = 64
    # elif args.dataset == 'lsunbedroom':
    #     img_size = 256

    # print(args.dataset, img_size)

    transform = transforms.Compose([transforms.ToTensor()])
    
    loss = torch.nn.MSELoss()
    
    sum, count = 0, 0

    input1_paths = os.listdir(args.targets_dir)
    for input1_path in input1_paths:
        input1_pil = Image.open(os.path.join(args.targets_dir, input1_path))
        input1_t = transform(input1_pil)
        input2_names = os.listdir(args.data_dir + '/' + os.path.basename(input1_path)[:-4])
        _sum, _count = 0, 0
        with torch.no_grad():
            for i in range(len(input2_names)):
                if 'jpg' in input2_names[i] or 'png' in input2_names[i]:
                    input2_path = os.path.join(args.data_dir, os.path.basename(input1_path)[:-4], input2_names[i])
                    input2_pil = Image.open(input2_path)
                    input2_t = transform(input2_pil)
                    
                    mseloss = loss(input1_t, input2_t)
                    sum += mseloss
                    count += 1
                    _sum += mseloss
                    _count += 1
            # print(input1_path, ' mse distance: ', (sum/count).item())
    print('total mse distance: ', (sum/count).item())


if __name__ == "__main__":
    main()