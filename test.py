import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
# from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from model.model import ResNet50
import torchvision.transforms as T
from data_loader.datasets.ood_dataset import ID_Dataset, Ood_Dataset
from torchvision.datasets import ImageFolder
from model.model import ResNet50
from utils.display_results import *
from torchvision.models import resnet50


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='resnet_ood_detection', help='Method name.')
# Loading details
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

# torch.manual_seed(1)
# np.random.seed(1)
# args.score = 'energy'
# mean and standard deviation of channels of CIFAR-10 images
basic_transform = T.Compose([T.ToTensor(), T.Resize((256, 256))])

fundus_test = '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/binary_fundus_test.csv'
uwf_test = '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/uwf_test.csv'
uwf_val =  '/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/data/uwf_val.csv'

# in-distribution data: fundus images
fundus_test = ID_Dataset(fundus_test, transform=basic_transform)

# out-of-distribution data: uwf, oct
ood_uwf_test = Ood_Dataset(uwf_test, transform=basic_transform)
ood_uwf_val = Ood_Dataset(uwf_val, transform=basic_transform)

# test_data
# test_data = torch.utils.data.ConcatDataset([fundus_test, ood_uwf_test, ood_uwf_val])
# test_data = ood_uwf_test
test_data = fundus_test

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
# net = ResNet50()
# net.load_state_dict(torch.load('./snapshots/oe_scratch/fundus_uwf_imgnet_resnet50_oe_scratch_epoch_19.pt'))
net = ResNet50()
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('/home/minsungkim/FUNDUS_RESEARCH/visualization/ood_detection/snapshots/oe_scratch/fix_sh_resnet50_oe_scratch_epoch_19.pt'))
# net.load_state_dict(torch.load('./snapshots/oe_scratch/fundus_uwf_imgnet_resnet50_oe_scratch_epoch_19.pt'))
net.eval()

if args.ngpu > 1:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    # net = torch.nn.DataParallel(net)

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5  # 논문에서 1:5로 실험 진행 (Out:1, In:5)
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


# TODO: in, out 이냐 판단 metric, in에서 성능 판단 metric 구현
def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:

                _score.append(np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                # preds = smax
                # preds = np.array(smax >= 1/13, dtype=float)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                # right_indices = [idx for idx, (i, j) in enumerate(zip(preds, targets)) if np.all(i == j) == True]
                wrong_indices = np.invert(right_indices)
                # wrong_indices = [idx for idx, (i, j) in enumerate(zip(preds, targets)) if np.all(i == j) != True]
                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else: 
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
                    # _right_score.append(smax[right_indices])
                    # _wrong_score.append(smax[wrong_indices])


    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

# args.use_xent = True
in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)
num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
# show_performance(wrong_score, right_score, method_name=args.method_name)
# show_performance(right_score, wrong_score, method_name=args.method_name)
# plot_auroc(wrong_score, right_score, fname='indist_binary_fundus_auroc')
# plot_aupr(wrong_score, right_score, fname="indist_binary_fundus_aupr")

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        # measures = get_measures(out_score, in_score)
        measures = get_measures(in_score, out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)


auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        # measures = get_measures(out_score, in_score)
        measures = get_measures(in_score, out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)
    
    plot_auroc(in_score, out_score, fname='ood_binary_fundus_auroc')
    plot_aupr(in_score, out_score, fname='ood_binary_fundus_aupr')

ood_test = Ood_Dataset(uwf_test, transform=basic_transform)
ood_val = Ood_Dataset(uwf_val, transform=basic_transform)

ood_data = torch.utils.data.ConcatDataset([ood_val, ood_test])

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=False,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nUWF Detection')
# args.use_xent=False
# args.use_xent=True
get_and_print_results(ood_loader)