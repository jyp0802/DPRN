#!/usr/bin/python2.7

import torch
from model import MSTCN_Trainer, DPN_Trainer, TSN_Trainer, DPRN_Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train_dpn')
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='1')
parser.add_argument('--gpu', default='0')
parser.add_argument('--epochs', default=0, type=int)
parser.add_argument('--mstcn_weights', default='')
parser.add_argument('--dpn_weights', default='')
parser.add_argument('--tsn_weights', default='')
parser.add_argument('--tune_weights', default='')
parser.add_argument('--save_dir', default='')
parser.add_argument('--f1', default=1, type=int)
parser.add_argument('--lr', type=float)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
print("Training on GPU #{}".format(args.gpu))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.autograd.set_detect_anomaly(True)

dpn_num_stages = 4
dpn_num_layers = 10
num_f_maps = 64
dpn_num_epochs = 400
if args.epochs > 0:
    dpn_num_epochs = args.epochs
dpn_tmse_lambda = 0.15
dpn_lr = 0.0005

tsn_num_stages = 3
md = "a" # "st" if shuffling truth values
tsn_kernel = 101
shuffle_window = 60
tsn_num_epochs = 100
if args.epochs > 0:
    tsn_num_epochs = args.epochs
tsn_lr = 0.0001

joint_lr = args.lr

features_dim = 2048
batch_size = 1

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

train_list_file = "data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
test_list_file = "data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "data/"+args.dataset+"/features/"
gt_path = "data/"+args.dataset+"/groundTruth/"

mapping_file = "data/"+args.dataset+"/mapping.txt"
save_dir = "ckpt/"+args.dataset+"/split_"+args.split+"/"+args.save_dir
mstcn_weights = "ckpt/"+args.dataset+"/split_"+args.split+"/"+args.mstcn_weights
dpn_weights = "trained_ckpt/"+args.dataset+"/"+args.dataset+"_DPN_split_"+args.split+".ckpt"
tsn_weights = "trained_ckpt/"+args.dataset+"/"+args.dataset+"_TRN_split_"+args.split+".ckpt"
results_dir = "results/"+args.dataset+"/split_"+args.split
 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[" ".join(a.split()[1:])] = int(a.split()[0])

num_classes = len(actions_dict)

if "train" in args.action:
    train_batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    train_batch_gen.read_data(train_list_file)
    val_batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    val_batch_gen.read_data(test_list_file)
    if args.action == "train_mstcn":
        trainer = MSTCN_Trainer(dpn_num_stages, dpn_num_layers, num_f_maps, features_dim, num_classes)
        trainer.train(save_dir, train_batch_gen, val_batch_gen, dpn_num_epochs, batch_size, dpn_lr, dpn_tmse_lambda, device, args.dataset)
    elif args.action == "train_dpn":
        trainer = DPN_Trainer(dpn_num_stages, dpn_num_layers, num_f_maps, features_dim, num_classes)
        trainer.train(save_dir, train_batch_gen, val_batch_gen, dpn_num_epochs, batch_size, dpn_lr, dpn_tmse_lambda, device, args.dataset)
    elif args.action == "train_tsn":
        trainer = TSN_Trainer(dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window)
        trainer.train(save_dir, train_batch_gen, val_batch_gen, tsn_num_epochs, batch_size, tsn_lr, device, dpn_weights, args.dataset, args.f1, md)
    elif args.action == "train_tune":
        trainer = DPRN_Trainer(dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window)
        trainer.train(save_dir, train_batch_gen, val_batch_gen, tsn_num_epochs, batch_size, joint_lr, device, dpn_weights, tsn_weights, dpn_tmse_lambda, args.dataset, args.f1, md)

if "predict" in args.action:
    if args.action == "predict_mstcn":
        trainer = MSTCN_Trainer(dpn_num_stages, dpn_num_layers, num_f_maps, features_dim, num_classes)
        trainer.predict(args.mstcn_weights, results_dir, features_path, test_list_file, actions_dict, device, sample_rate)
    elif args.action == "predict_dpn":
        trainer = DPN_Trainer(dpn_num_stages, dpn_num_layers, num_f_maps, features_dim, num_classes)
        trainer.predict(dpn_weights, results_dir, features_path, test_list_file, actions_dict, device, sample_rate)
    elif args.action == "predict_tsn":
        trainer = TSN_Trainer(dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window)
        trainer.predict(dpn_weights, tsn_weights, results_dir, features_path, test_list_file, actions_dict, device, sample_rate)
    elif args.action == "predict_tune":
        trainer = DPRN_Trainer(dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window)
        trainer.predict(args.tune_weights, results_dir, features_path, test_list_file, actions_dict, device, sample_rate)
