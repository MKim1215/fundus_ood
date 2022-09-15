import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torchvision.datasets as datasets
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from data_loader.datasets.ood_dataset import Ood_Dataset
from data_loader.datasets.ood_dataset import ID_Dataset
from model.model import ResNet50
from itertools import cycle

# go through rigamaroo to do ...utils.display_results import show_performance
# if __package__ is None:
#     import sys
#     from os import path

#     sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
#     from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Trains a OOD Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--calibration', '-c', action='store_true',
#                     help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options

parser.add_argument('--dataset', type=str, default='fundus_uwf')
parser.add_argument('--model', '-m', type=str, default='resnet50', help='Choose architecture.')
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=4, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = T.Compose([T.Resize((256, 256)), T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                               T.ToTensor(), T.Normalize(mean, std)])
test_transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean, std)])

fundus_train = '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/binary_fundus_train.csv'
fundus_test = '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/binary_fundus_test.csv'
fundus_val = '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/binary_fundus_val.csv'
uwf_train = '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/uwf_train.csv'
uwf_test = '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/uwf_test.csv'
uwf_val = '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/uwf_val.csv'


# in-distribution data: fundus images
fundus_train = ID_Dataset(fundus_train, transform=T.Compose([T.ToTensor(), T.Resize((256, 256))]))
fundus_test = ID_Dataset(fundus_test, transform=T.Compose([T.ToTensor(), T.Resize((256, 256))]))
fundus_val = ID_Dataset(fundus_val, transform=T.Compose([T.ToTensor(), T.Resize((256, 256))]))

ood_train = Ood_Dataset(uwf_train, transform=T.Compose([T.ToTensor(), T.Resize((256, 256))]))
ood_test = Ood_Dataset(uwf_test, transform=T.Compose([T.ToTensor(), T.Resize((256, 256))]))
ood_val = Ood_Dataset(uwf_val, transform=T.Compose([T.ToTensor(), T.Resize((256, 256))]))

total_train = torch.utils.data.ConcatDataset([fundus_train, ood_train])
# total_val = torch.utils.data.ConcatDataset([fundus_test, ood_val])


# args.dataset = 'ood_dataset'
# args.model = 'resnet50'

# calib_indicator = ''
# if args.calibration:
#     train_data, val_data = validation_split(ood_train_data, val_share=0.1)
#     calib_indicator = '_calib'

train_loader = DataLoader(
    total_train,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)


test_loader = torch.utils.data.DataLoader(
    fundus_val,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

fundus_loader = torch.utils.data.DataLoader(
    fundus_val,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

uwf_loader = torch.utils.data.DataLoader(
    ood_val,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
net = ResNet50()

start_epoch = 0

# Restore model if desired
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + calib_indicator + '_' + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

if args.ngpu > 1:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    # net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    net = torch.nn.DataParallel(net)


if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////

def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for data, target in tqdm(train_loader):
        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # backward
        # scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    scheduler.step()
    state['train_loss'] = loss_avg

# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['val_loss'] = loss_avg / len(test_loader)
    state['val_accuracy'] = correct / len(test_loader.dataset)

        # for in_set, out_set in zip(fundus_loader, cycle(uwf_loader)):
        #     total_data = torch.cat((in_set[0], out_set[0]), 0)
        #     total_label = torch.cat((in_set[1], out_set[1]))
        #     data, target = total_data.cuda(), total_label.cuda()

        #     # forward
        #     output = net(data)
        #     loss = F.cross_entropy(output, target)

        #     # accuracy
        #     pred = output.data.max(1)[1]
        #     correct += pred.eq(target.data).sum().item()

        #     # test loss average
        #     loss_avg += float(loss.data)

    # state['test_loss'] = loss_avg / len(test_loader)
    # state['test_accuracy'] = correct / len(test_loader.dataset)
    # state['val_loss'] = loss_avg / (len(fundus_loader) * 2)
    # state['val_accuracy'] = correct / (len(fundus_loader.dataset) * 2)

if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

# with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
#                                   '_baseline_training_results.csv'), 'w') as f:
#     f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

with open(os.path.join(args.save, args.dataset + '_' + args.model +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,val_loss,test_error(%)\n')


print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    # torch.save(net.state_dict(),
    #            os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
    #                         '_baseline_epoch_' + str(epoch) + '.pt'))
    # # Let us not waste space and delete the previous model
    # prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
    #                          '_baseline_epoch_' + str(epoch - 1) + '.pt')

    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + '_' + args.model +
                            '_baseline_epoch_' + str(epoch) + '.pt'))

    prev_path = os.path.join(args.save, args.dataset + '_' + args.model +
                             '_baseline_epoch_' + str(epoch - 1) + '.pt')

    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    # with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
    #                                   '_baseline_training_results.csv'), 'a') as f:
    #     f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
    #         (epoch + 1),
    #         time.time() - begin_epoch,
    #         state['train_loss'],
    #         state['test_loss'],
    #         100 - 100. * state['test_accuracy'],
    #     ))

    with open(os.path.join(args.save, args.dataset + '_' + args.model +
                                      '_baseline_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['val_loss'],
            100 - 100. * state['val_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['val_loss'],
        100 - 100. * state['val_accuracy'])
    )