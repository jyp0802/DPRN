#!/usr/bin/python2.7
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import random
import sys
import metrics as metrics


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DPSN(nn.Module):
    def __init__(self, dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window):
        super(DPSN, self).__init__()
        self.dpn = MultiStageDPN(dpn_num_stages, dpn_num_layers, num_f_maps, features_dim, num_classes)
        self.tsn = MultiStageTSN(tsn_num_stages, tsn_kernel, num_f_maps, num_classes, num_classes)
        self.shuffle_window = shuffle_window

    def forward(self, x, mask, mode, target, md):
        dpn_out = self.dpn(x, mask)

        if mode == "train":
            tsn_in = shuffle_input(dpn_out[-1], self.shuffle_window, target, md)
        else:
            tsn_in = dpn_out[-1]

        tsn_out = self.tsn(tsn_in, mask)
        fin_out = torch.cat((dpn_out, tsn_out), dim=0)

        return fin_out, tsn_in


class MultiStageDPN(nn.Module):
    def __init__(self, dpn_num_stages, num_layers, num_f_maps, features_dim, num_classes):
        super(MultiStageDPN, self).__init__()
        self.stage1 = OldSingleStageDPN(num_layers, num_f_maps, features_dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(OldSingleStageDPN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(dpn_num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageDPN(nn.Module):
    def __init__(self, num_layers, num_f_maps, features_dim, num_classes):
        super(SingleStageDPN, self).__init__()
        self.conv_1x1 = nn.Conv1d(features_dim, num_f_maps, 1)
        self.dl_layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        #
        self.depthwise = nn.Conv1d(num_f_maps*(num_layers+1), num_f_maps*(num_layers+1) * 2, kernel_size=3, padding=1, groups=num_f_maps*(num_layers+1)//2)
        self.fo_out0 = nn.Conv1d(num_f_maps*(num_layers+1), num_f_maps*(num_layers+1), 1)
        #
        self.conv_reduce = nn.Conv1d((num_layers+1)*num_f_maps, num_f_maps, 1)

        # self.d_conv_in = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps, num_f_maps, 1)) for i in range(num_layers)])
        # self.d_conv_down = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps, num_f_maps, 1)) for i in range(num_layers-1)])
        # self.u_conv_in = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps, num_f_maps, 1)) for i in range(num_layers)])
        # self.u_conv_up = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps, num_f_maps, 1)) for i in range(num_layers-1)])
        
        self.d_conv_in = nn.ModuleList([nn.Conv1d(num_f_maps, num_f_maps, 1) for i in range(num_layers)])
        self.d_conv_down = nn.ModuleList([nn.Conv1d(num_f_maps, num_f_maps, 1) for i in range(num_layers-1)])
        self.u_conv_in = nn.ModuleList([nn.Conv1d(num_f_maps, num_f_maps, 1) for i in range(num_layers)])
        self.u_conv_up = nn.ModuleList([nn.Conv1d(num_f_maps, num_f_maps, 1) for i in range(num_layers-1)])

        self.nl = [i for i in range(num_layers+1)]
        self.num_layers = num_layers

    def forward(self, x, mask):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        out = self.conv_1x1(x)
        out_0 = out
        
        d_in = []
        for i in range(self.num_layers):
            out = self.dl_layers[i](out, mask)
            d_in.append(self.d_conv_in[i](out))

        d_out = [d_in[-1]]
        for i in reversed(range(self.num_layers-1)):
            d_out.append(d_in[i] + self.d_conv_down[i](d_in[i+1]))
        d_out.reverse()

        u_in = []
        for i in range(self.num_layers):
            u_in.append(self.u_conv_in[i](d_out[i]))

        u_out = []
        for i in reversed(range(self.num_layers-1)):
            u_out.append(u_in[i+1] + self.u_conv_up[i](u_in[i]))
        u_out.append(u_in[0])
        u_out.reverse()

        out_cat = torch.cat(([out_0] + u_out), 1)

        out = self.conv_reduce(out_cat)

        final_out = (self.conv_out(out)) * mask[:, 0:1, :]

        return final_out

class OldSingleStageDPN(nn.Module):
    def __init__(self, num_layers, num_f_maps, features_dim, num_classes):
        super(OldSingleStageDPN, self).__init__()
        self.conv_1x1 = nn.Conv1d(features_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fo_out = nn.Conv1d(num_f_maps*(num_layers+1), num_f_maps, 1) # NOT DEPTHWISE
        self.nl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.c1 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c2 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c3 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c4 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c5 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c6 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c7 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c8 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c9 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.c10 = nn.Conv1d(num_f_maps, num_f_maps, 1)

        self.ca1 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.ca2 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.ca3 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.ca4 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.ca5 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.ca6 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.ca7 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.ca8 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.ca9 = nn.Conv1d(num_f_maps, num_f_maps, 1)

        self.d1 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d2 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d3 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d4 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d5 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d6 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d7 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d8 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d9 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.d10 = nn.Conv1d(num_f_maps, num_f_maps, 1)

        self.da2 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.da3 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.da4 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.da5 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.da6 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.da7 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.da8 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.da9 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.da10 = nn.Conv1d(num_f_maps, num_f_maps, 1)


    def forward(self, x, mask):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        out = self.conv_1x1(x)
        oc = out
        oc2 = out
        
        out1 = self.layers[0](out, mask)
        out1_ = self.c1(out1)

        out2 = self.layers[1](out1, mask)
        out2_ = self.c2(out2)

        out3 = self.layers[2](out2, mask)
        out3_ = self.c3(out3)

        out4 = self.layers[3](out3, mask)
        out4_ = self.c4(out4)

        out5 = self.layers[4](out4, mask)
        out5_ = self.c5(out5)

        out6 = self.layers[5](out5, mask)
        out6_ = self.c6(out6)

        out7 = self.layers[6](out6, mask)
        out7_ = self.c7(out7)

        out8 = self.layers[7](out7, mask)
        out8_ = self.c8(out8)

        out9 = self.layers[8](out8, mask)
        out9_ = self.c9(out9)

        out10 = self.layers[9](out9, mask)
        out10_ = self.c10(out10)

        out9_ = self.ca9(out10_) + out9_
        out8_ = self.ca8(out9_) + out8_
        out7_ = self.ca7(out8_) + out7_
        out6_ = self.ca6(out7_) + out6_
        out5_ = self.ca5(out6_) + out5_
        out4_ = self.ca4(out5_) + out4_
        out3_ = self.ca3(out4_) + out3_
        out2_ = self.ca2(out3_) + out2_
        out1_ = self.ca1(out2_) + out1_

        out1__ = self.d1(out1_)
        out2__ = self.d2(out2_)
        out3__ = self.d3(out3_)
        out4__ = self.d4(out4_)
        out5__ = self.d5(out5_)
        out6__ = self.d6(out6_)
        out7__ = self.d7(out7_)
        out8__ = self.d8(out8_)
        out9__ = self.d9(out9_)
        out10__ = self.d10(out10_)

        out10__ = self.da10(out9__) + out10__
        out9__ = self.da9(out8__) + out9__
        out8__ = self.da8(out7__) + out8__
        out7__ = self.da7(out6__) + out7__
        out6__ = self.da6(out5__) + out6__
        out5__ = self.da5(out4__) + out5__
        out4__ = self.da4(out3__) + out4__
        out3__ = self.da3(out2__) + out3__
        out2__ = self.da2(out1__) + out2__

        O_out = torch.cat((oc, out1__, out2__, out3__, out4__, out5__, out6__, out7__, out8__, out9__, out10__), 1)

        fo = self.fo_out(O_out)

        final_out = (self.conv_out(fo)) * mask[:, 0:1, :]
        return final_out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        # out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class MultiStageTSN(nn.Module):
    def __init__(self, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes):
        super(MultiStageTSN, self).__init__()
        self.stage1 = SingleStageTSN(tsn_kernel, num_f_maps, num_classes, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageTSN(tsn_kernel, num_f_maps, num_classes, num_classes)) for s in range(tsn_num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for stage in self.stages:
            out = stage(out+x, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageTSN(nn.Module):
    def __init__(self, tsn_kernel, num_f_maps, features_dim, num_classes):
        super(SingleStageTSN, self).__init__()
        self.conv_1x1 = nn.Conv1d(features_dim, num_f_maps, kernel_size=3, padding=1)
        self.conv_down_1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=tsn_kernel, stride=2, padding=tsn_kernel//2)
        self.pool1d = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.conv_mid = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=3, padding=1)
        self.conv_up_1 = nn.ConvTranspose1d(num_f_maps, num_f_maps, kernel_size=tsn_kernel, stride=2, padding=tsn_kernel//2)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        out = F.relu(out)

        out = self.conv_down_1(out)
        temp_size = out.shape[2]
        out = self.pool1d(out)

        out = self.conv_mid(out)
        out = F.relu(out)

        out = F.interpolate(out, scale_factor=2)
        if out.shape[2] < temp_size:
            out = F.pad(out, (0,temp_size-out.shape[2]), "constant", 0)

        out = self.dropout(out)

        out = self.conv_up_1(out)
        if out.shape[2] < x.shape[2]:
            out = F.pad(out, (0,x.shape[2]-out.shape[2]), "constant", 0)

        out = out[:,:,:x.shape[2]]
        out = self.conv_out(out)
        return out


def shuffle_input(out, shuffle_window, target, md):
    if md != "st":
        _, c = torch.max(out.data, 1)
        c = c.squeeze()
        orig_out = out.detach()

        l = out.shape[2]
    else:
        c = target.squeeze()
        orig_out = out.detach()

        l = target.shape[1]


    cur = c[0]
    bd = []
    for i in range(c.shape[0]):
        if c[i] != cur:
            bd.append(i)
            cur = c[i]

    for b in bd:
        lw, rw = torch.randint(shuffle_window//10, shuffle_window//2, (2,))
        lw, rw = int(lw.item()), int(rw.item())
        rndi = int(torch.randint(0, l-lw-rw, (1,)).item())
        if b-lw < 0 or b+rw >= l:
            continue
        out[:,:,b-lw:b+rw] = orig_out[:,:,rndi:rndi+lw+rw]

    for b in bd:
        seglen = int(torch.randint(shuffle_window//5, shuffle_window, (1,)).item())
        fi, ri = torch.randint(0, l-seglen, (2,))
        fi, ri = int(fi.item()), int(ri.item())
        out[:,:,fi:fi+seglen] = orig_out[:,:,ri:ri+seglen]

    return out


def get_weight(window, x):
    _, x = torch.max(x.data, 1)
    x = x.squeeze()
    padding = window//2

    cons = [1]*padding
    cur = -1
    cnt = 0
    for t in range(x.shape[0]):
        if x[t] != cur:
            cnt += 1
            cur = x[t]
        cons.append(cnt)
    cons += [cons[-1]]*padding

    w = []
    for i in range(padding, x.shape[0]+padding):
        w.append(cons[i+padding]-cons[i-padding]+1)
    
    w = torch.tensor(w).unsqueeze(0).float()
    return w


def wmse(weight, pred, target):
    return weight * (pred - target) ** 2


class MSTCN_Trainer:
    def __init__(self, num_stages, num_layers, num_f_maps, features_dim, num_classes):
        self.model = MultiStageModel(num_stages, num_layers, num_f_maps, features_dim, num_classes)

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("MSTCN", pytorch_total_params)
        print("num_stages, num_layers, num_f_maps, features_dim, num_classes: ", num_stages, num_layers, num_f_maps, features_dim, num_classes)

    def train(self, save_dir, batch_gen, test_batch_gen, num_epochs, batch_size, learning_rate, tmse_lambda, device, dataset):
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        max_epoch = 0
        max_acc = 0

        v_max_epoch = 0
        v_max_acc = 0

        self.model.train()

        for epoch in range(num_epochs):
            #### TRAIN ####
            
            total = 0.0
            acc = 0.0

            e_loss = 0.0
            e_loss_ce = 0
            e_loss_tmse = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss_ce = 0
                loss_tmse = 0
                for p in predictions:
                    loss_ce += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss_tmse += tmse_lambda*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                e_loss_ce += loss_ce.item()
                e_loss_tmse += loss_tmse.item()

                loss = loss_ce + loss_tmse

                e_loss += loss.item()
                loss.backward()
                optimizer.step()

                total += torch.sum(mask[:, 0, :]).item()
                _, predicted_Y = torch.max(predictions[-1].data, 1)
                acc += metrics.get_correct(predicted_Y, batch_target)

            batch_gen.reset()

            epoch_size = len(batch_gen.list_of_examples)
            acc /= total
            print("[epoch {}]: loss = {:.4f}, loss_ce = {:.4f}, loss_tmse = {:.4f}, acc = {:.4f}"\
                    .format(epoch, e_loss/epoch_size, e_loss_ce/epoch_size, e_loss_tmse/epoch_size, acc), end='')


            #### VALIDATION ####
            self.model.eval()

            total = 0.0
            acc = 0.0

            with torch.no_grad():
                while test_batch_gen.has_next():
                    batch_input, batch_target, mask = test_batch_gen.next_batch(batch_size)
                    batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                    predictions = self.model(batch_input, mask)

                    total += torch.sum(mask[:, 0, :]).item()
                    _, predicted_Y = torch.max(predictions[-1].data, 1)
                    acc += metrics.get_correct(predicted_Y, batch_target)

                test_batch_gen.reset()

            epoch_size = len(test_batch_gen.list_of_examples)
            acc /= total
            if acc > v_max_acc:
                v_max_epoch = epoch
                v_max_acc = acc
                if acc > 0.75:
                    torch.save(self.model.state_dict(), "{}/MSTCN_Epoch_{}_Acc_{:.4f}.ckpt".format(save_dir, epoch, acc))
            print("\t VAL: acc = {:.4f}, max_acc = {:.4f}, max_epoch = {}"\
                    .format(acc, v_max_acc, v_max_epoch))

            torch.save(self.model.state_dict(), "{}/Epoch_{}_Acc_{:.4f}.ckpt".format(save_dir, epoch, acc))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, actions_dict, device, sample_rate):
        state = torch.load(model_dir)
        self.model.load_state_dict(state)
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()



class DPN_Trainer:
    def __init__(self, num_stages, num_layers, num_f_maps, features_dim, num_classes):
        self.model = MultiStageDPN(num_stages, num_layers, num_f_maps, features_dim, num_classes)

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("DPN", pytorch_total_params)
        print("num_stages, num_layers, num_f_maps, features_dim, num_classes: ", num_stages, num_layers, num_f_maps, features_dim, num_classes)

    def train(self, save_dir, batch_gen, test_batch_gen, num_epochs, batch_size, learning_rate, tmse_lambda, device, dataset):
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        max_epoch = 0
        max_acc = 0

        v_max_epoch = 0
        v_max_acc = 0
        
        self.model.train()

        for epoch in range(num_epochs):
            #### TRAIN ####

            total = 0.0
            acc = 0.0

            e_loss = 0.0
            e_loss_ce = 0
            e_loss_tmse = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss_ce = 0
                loss_tmse = 0
                for p in predictions:
                    loss_ce += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss_tmse += tmse_lambda*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                e_loss_ce += loss_ce.item()
                e_loss_tmse += loss_tmse.item()

                loss = loss_ce + loss_tmse

                e_loss += loss.item()
                loss.backward()
                optimizer.step()

                total += torch.sum(mask[:, 0, :]).item()
                _, predicted_Y = torch.max(predictions[-1].data, 1)
                acc += metrics.get_correct(predicted_Y, batch_target)

            batch_gen.reset()

            epoch_size = len(batch_gen.list_of_examples)
            acc /= total
            print("[epoch {}]: loss = {:.4f}, loss_ce = {:.4f}, loss_tmse = {:.4f}, acc = {:.4f}"\
                    .format(epoch, e_loss/epoch_size, e_loss_ce/epoch_size, e_loss_tmse/epoch_size, acc), end='')

            #### VALIDATION ####
            self.model.eval()

            total = 0.0
            acc = 0.0

            with torch.no_grad():
                while test_batch_gen.has_next():
                    batch_input, batch_target, mask = test_batch_gen.next_batch(batch_size)
                    batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                    predictions = self.model(batch_input, mask)

                    total += torch.sum(mask[:, 0, :]).item()
                    _, predicted_Y = torch.max(predictions[-1].data, 1)
                    acc += metrics.get_correct(predicted_Y, batch_target)

                test_batch_gen.reset()

            epoch_size = len(test_batch_gen.list_of_examples)
            acc /= total
            if acc > v_max_acc:
                v_max_epoch = epoch
                v_max_acc = acc
                if acc > 0.75:
                    torch.save(self.model.state_dict(), "{}/DPN_Epoch_{}_Acc_{:.4f}.ckpt".format(save_dir, epoch, acc))
            print("\t VAL: acc = {:.4f}, max_acc = {:.4f}, max_epoch = {}"\
                    .format(acc, v_max_acc, v_max_epoch))
                    

    def predict(self, dpn_weights, results_dir, features_path, vid_list_file, actions_dict, device, sample_rate):        
        self.model.to(device)
        self.model.load_state_dict(torch.load(dpn_weights))

        with torch.no_grad():
            self.model.eval()
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()


class TSN_Trainer:
    def __init__(self, dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window):
        self.model = DPSN(dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window)

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        self.tsn_num_stages = tsn_num_stages
        self.tsn_kernel = tsn_kernel
        
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("DPRN", pytorch_total_params)

        pytorch_total_params = sum(p.numel() for p in self.model.dpn.parameters() if p.requires_grad)
        print("DPN", pytorch_total_params)

        pytorch_total_params = sum(p.numel() for p in self.model.tsn.parameters() if p.requires_grad)
        print("TRN", pytorch_total_params)

    def train(self, save_dir, batch_gen, test_batch_gen, num_epochs, batch_size, learning_rate, device, dpn_weights, dataset, f1_every, md):
        self.model.to(device)
        self.model.dpn.load_state_dict(torch.load(dpn_weights))

        if dataset == '50salads':
            BGCLASS = [17,18]
        elif dataset == 'breakfast':
            BGCLASS = []
        elif dataset == 'gtea':
            BGCLASS = [10]
        else:
            print('What dataset is this?')
            exit()

        optimizer = optim.Adam(self.model.tsn.parameters(), lr=learning_rate)

        overlap = [.1, .25, .5]

        max_epoch = 0
        max_acc = 0
        max_edit = 0
        max_f1 = [0,0,0]

        v_max_epoch = 0
        v_max_acc = 0
        v_max_edit = 0
        v_max_f1 = [0,0,0]

        for epoch in range(num_epochs):
            #### TRAIN ####
            self.model.train()

            total = 0.0
            acc = 0.0
            acc_back = 0.0

            edit = 0.0
            f1 = [0,0,0]
            tpfpfn = np.zeros((3, 3))

            e_loss = 0.0
            e_loss_ce = 0
            e_loss_tmse = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions, dpn_out = self.model(batch_input, mask, "train", batch_target, md)

                w = get_weight(self.tsn_kernel, dpn_out.detach()).to(device)
                w = w / torch.max(w.data, 1)[0]

                loss_ce = 0
                loss_tmse = 0                
                for p in predictions[-self.tsn_num_stages:]:
                    loss_ce += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss_tmse += 0.5*torch.mean(wmse(w[:, 1:], F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1))*mask[:, :, 1:])
                    loss_tmse += 0.5*torch.mean(wmse(w[:, :-1], F.log_softmax(p[:, :, :-1], dim=1), F.log_softmax(p.detach()[:, :, 1:], dim=1))*mask[:, :, :-1])

                e_loss_ce += loss_ce.item()
                e_loss_tmse += loss_tmse.item()

                loss = loss_ce + loss_tmse

                e_loss += loss.item()
                loss.backward()
                optimizer.step()

                total += torch.sum(mask[:, 0, :]).item()
                _, predicted_Y = torch.max(predictions[-1].data, 1)
                acc += metrics.get_correct(predicted_Y, batch_target)
                _, predicted_Y_back = torch.max(predictions[-self.tsn_num_stages-1].data, 1)
                acc_back += metrics.get_correct(predicted_Y_back, batch_target)

            batch_gen.reset()

            epoch_size = len(batch_gen.list_of_examples)
            acc_back /= total
            acc /= total
            print("[epoch {}]: loss = {:.4f}, loss_ce = {:.4f}, loss_tmse = {:.4f}, acc_back = {:.4f}, acc = {:.4f}"\
                    .format(epoch, e_loss/epoch_size, e_loss_ce/epoch_size, e_loss_tmse/epoch_size, acc_back, acc), end='')

            #### VALIDATION ####
            self.model.eval()

            total = 0.0
            acc = 0.0
            acc_back = 0.0
            
            edit = 0.0
            f1 = [0,0,0]
            tpfpfn = np.zeros((3, 3))

            with torch.no_grad():
                while test_batch_gen.has_next():
                    batch_input, batch_target, mask = test_batch_gen.next_batch(batch_size)
                    batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                    predictions, _ = self.model(batch_input, mask, "test", "", md)

                    total += torch.sum(mask[:, 0, :]).item()
                    _, predicted_Y = torch.max(predictions[-1].data, 1)
                    acc += metrics.get_correct(predicted_Y, batch_target)
                    _, predicted_Y_back = torch.max(predictions[-self.tsn_num_stages-1].data, 1)
                    acc_back += metrics.get_correct(predicted_Y_back, batch_target)

                    if (epoch+1) % f1_every == 0:
                        edit += metrics.get_edit_score(predicted_Y, batch_target, bg_class = BGCLASS)
                        for s in range(len(overlap)):
                            tpfpfn[s] += np.array(metrics.get_f_score(predicted_Y, batch_target, overlap[s], bg_class = BGCLASS))

                test_batch_gen.reset()

            epoch_size = len(test_batch_gen.list_of_examples)
            acc /= total
            acc_back /= total
            if acc > v_max_acc:
                v_max_epoch = epoch
                v_max_acc = acc
            print("\t VAL: acc = {:.4f}, acc_back = {:.4f}, max_acc = {:.4f}, max_epoch = {}"\
                    .format(acc, acc_back, v_max_acc, v_max_epoch))
            if (epoch+1) % f1_every == 0:
                for s in range(len(overlap)):
                    precision = tpfpfn[s,0] / float(tpfpfn[s,0]+tpfpfn[s,1])
                    recall = tpfpfn[s,0] / float(tpfpfn[s,0]+tpfpfn[s,2])
                    f1[s] = np.nan_to_num(2.0 * (precision*recall) / (precision+recall))*100
                    v_max_f1[s] = max(v_max_f1[s], f1[s])

                edit /= epoch_size
                v_max_edit = max(v_max_edit, edit)
                print("\t VAL: acc = {:.4f}, edit = {:.4f}, f1 = {}, {}, {} , max_edit = {:.4f}, max_f1 = {}"\
                        .format(acc, edit, f1[0], f1[1], f1[2], v_max_edit, v_max_f1))

            torch.save(self.model.tsn.state_dict(), "{}/Epoch_{}.ckpt".format(save_dir, epoch, edit))


    def predict(self, dpn_weights, tsn_weights, results_dir, features_path, vid_list_file, actions_dict, device, sample_rate):
        
        self.model.to(device)
        self.model.dpn.load_state_dict(torch.load(dpn_weights))
        self.model.tsn.load_state_dict(torch.load(tsn_weights))

        with torch.no_grad():
            self.model.eval()
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, _ = self.model(input_x, torch.ones(input_x.size(), device=device), "test", None, None)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()


class DPRN_Trainer:
    def __init__(self, dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window):
        self.model = DPSN(dpn_num_stages, dpn_num_layers, tsn_num_stages, tsn_kernel, num_f_maps, features_dim, num_classes, shuffle_window)

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        self.tsn_num_stages = tsn_num_stages
        self.tsn_kernel = tsn_kernel

    def train(self, save_dir, batch_gen, test_batch_gen, num_epochs, batch_size, learning_rate, device, dpn_weights, tsn_weights, tmse_lambda, dataset, f1_every, md):
        self.model.to(device)
        self.model.dpn.load_state_dict(torch.load(dpn_weights))
        self.model.tsn.load_state_dict(torch.load(tsn_weights))

        if dataset == '50salads':
            BGCLASS = [17,18]
        elif dataset == 'breakfast':
            BGCLASS = []
        elif dataset == 'gtea':
            BGCLASS = [10]
        else:
            print('What dataset is this?')
            exit()

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # optimizer = optim.Adam([
        #     {"params": self.model.dpn.parameters(), "lr": 0.0001},
        #     {"params": self.model.tsn.parameters()},
        # ], lr=0.00001)

        overlap = [.1, .25, .5]

        max_epoch = 0
        max_acc = 0
        max_edit = 0
        max_f1 = [0,0,0]

        v_max_epoch = 0
        v_max_acc = 0
        v_max_edit = 0
        v_max_f1 = [0,0,0]

        for epoch in range(num_epochs):
            #### TRAIN ####
            self.model.train()

            total = 0.0
            acc = 0.0
            acc_back = 0.0

            edit = 0.0
            f1 = [0,0,0]
            tpfpfn = np.zeros((3, 3))

            e_loss = 0.0
            e_loss_ce = 0
            e_loss_tmse = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions, dpn_out = self.model(batch_input, mask, "train", batch_target, md)

                w = get_weight(self.tsn_kernel, dpn_out.detach()).to(device)
                w = w / torch.max(w.data, 1)[0]

                loss_ce = 0
                loss_tmse = 0
                
                loss_ce += self.ce(dpn_out.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                # loss_tmse += tmse_lambda*torch.mean(torch.clamp(self.mse(F.log_softmax(dpn_out[:, :, 1:], dim=1), F.log_softmax(dpn_out.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                for p in predictions[-self.tsn_num_stages:]:
                # for p in predictions[-1:]:
                    loss_ce += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss_tmse += 0.5*torch.mean(wmse(w[:, 1:], F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1))*mask[:, :, 1:])
                    loss_tmse += 0.5*torch.mean(wmse(w[:, :-1], F.log_softmax(p[:, :, :-1], dim=1), F.log_softmax(p.detach()[:, :, 1:], dim=1))*mask[:, :, :-1])

                e_loss_ce += loss_ce.item()
                e_loss_tmse += loss_tmse.item()

                loss = loss_ce + loss_tmse

                e_loss += loss.item()
                loss.backward()
                optimizer.step()

                total += torch.sum(mask[:, 0, :]).item()
                _, predicted_Y = torch.max(predictions[-1].data, 1)
                acc += metrics.get_correct(predicted_Y, batch_target)
                _, predicted_Y_back = torch.max(predictions[-self.tsn_num_stages-1].data, 1)
                acc_back += metrics.get_correct(predicted_Y_back, batch_target)

            batch_gen.reset()

            epoch_size = len(batch_gen.list_of_examples)
            acc_back /= total
            acc /= total
            print("[epoch {}]: loss = {:.4f}, loss_ce = {:.4f}, loss_tmse = {:.4f}, acc_back = {:.4f}, acc = {:.4f}"\
                    .format(epoch, e_loss/epoch_size, e_loss_ce/epoch_size, e_loss_tmse/epoch_size, acc_back, acc), end='')

            #### VALIDATION ####
            self.model.eval()

            total = 0.0
            acc = 0.0
            acc_back = 0.0
            
            edit = 0.0
            f1 = [0,0,0]
            tpfpfn = np.zeros((3, 3))

            with torch.no_grad():
                while test_batch_gen.has_next():
                    batch_input, batch_target, mask = test_batch_gen.next_batch(batch_size)
                    batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                    predictions, _ = self.model(batch_input, mask, "test", "", md)

                    total += torch.sum(mask[:, 0, :]).item()
                    _, predicted_Y = torch.max(predictions[-1].data, 1)
                    acc += metrics.get_correct(predicted_Y, batch_target)
                    _, predicted_Y_back = torch.max(predictions[-self.tsn_num_stages-1].data, 1)
                    acc_back += metrics.get_correct(predicted_Y_back, batch_target)

                    if (epoch+1) % f1_every == 0:
                        edit += metrics.get_edit_score(predicted_Y, batch_target, bg_class = BGCLASS)
                        for s in range(len(overlap)):
                            tpfpfn[s] += np.array(metrics.get_f_score(predicted_Y, batch_target, overlap[s], bg_class = BGCLASS))

                test_batch_gen.reset()

            epoch_size = len(test_batch_gen.list_of_examples)
            acc /= total
            acc_back /= total
            if acc > v_max_acc:
                v_max_epoch = epoch
                v_max_acc = acc
            print("\t VAL: acc = {:.4f}, acc_back = {:.4f}, max_acc = {:.4f}, max_epoch = {}"\
                    .format(acc, acc_back, v_max_acc, v_max_epoch))
            if (epoch+1) % f1_every == 0:
                for s in range(len(overlap)):
                    precision = tpfpfn[s,0] / float(tpfpfn[s,0]+tpfpfn[s,1])
                    recall = tpfpfn[s,0] / float(tpfpfn[s,0]+tpfpfn[s,2])
                    f1[s] = np.nan_to_num(2.0 * (precision*recall) / (precision+recall))*100
                    v_max_f1[s] = max(v_max_f1[s], f1[s])

                edit /= epoch_size
                v_max_edit = max(v_max_edit, edit)
                print("\t VAL: acc = {:.4f}, edit = {:.4f}, f1 = {}, {}, {} , max_edit = {:.4f}, max_f1 = {}"\
                        .format(acc, edit, f1[0], f1[1], f1[2], v_max_edit, v_max_f1))

            torch.save(self.model.state_dict(), "{}/Epoch_{}_{:.4f}.ckpt".format(save_dir, epoch, acc))


    def predict(self, tune_weights, results_dir, features_path, vid_list_file, actions_dict, device, sample_rate):
        
        self.model.to(device)
        self.model.load_state_dict(torch.load(tune_weights))

        with torch.no_grad():
            self.model.eval()
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, _ = self.model(input_x, torch.ones(input_x.size(), device=device), "test", None, None)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
