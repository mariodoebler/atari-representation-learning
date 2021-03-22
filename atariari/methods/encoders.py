import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init
import time
from atariari.benchmark.utils import download_run
from atariari.benchmark.episodes import checkpointed_steps_full_sorted
import os

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels, args):
        super(ImpalaCNN, self).__init__()
        self.hidden_size = args.feature_size
        self.depths = [16, 32, 32, 32]
        self.downsample = not args.no_downsample
        self.layer1 = self._make_layer(input_channels, self.depths[0])
        self.layer2 = self._make_layer(self.depths[0], self.depths[1])
        self.layer3 = self._make_layer(self.depths[1], self.depths[2])
        self.layer4 = self._make_layer(self.depths[2], self.depths[3])
        if self.downsample:
            self.final_conv_size = 32 * 9 * 9
        else:
            self.final_conv_size = 32 * 12 * 9
        self.final_linear = nn.Linear(self.final_conv_size, self.hidden_size)
        self.flatten = Flatten()
        self.train()

    def _make_layer(self, in_channels, depth):
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward(self, inputs, fmaps=False):
        f5 = self.layer3(self.layer2(self.layer1(inputs)))

        if not self.downsample:
            out = self.layer4(f5)
        else:
            out = f5

        out = F.relu(self.final_linear(self.flatten(out)))

        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }

        return out

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class NatureCNN(nn.Module):

    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample if args.no_downsample else False
        self.less_dense = args.less_dense if args.less_dense else False
        self.more_dense = args.more_dense if args.more_dense else False
        self.more_spatial_dim = args.more_spatial_dim if args.more_spatial_dim else False
        self.input_channels = input_channels
        self.end_with_relu = args.end_with_relu if args.end_with_relu else False
        self.args = args
        if type(args) == dict:
            self.input_110_84 = args.get("input_110_84", False)
        else:
            try:
                self.input_110_84 = args['input_110_84']
            except:
                self.input_110_84 = False
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        if self.downsample:
            if self.more_dense:
                last_nr_conv_filters = 70
                self.final_conv_shape = (last_nr_conv_filters, 7, 7)
                self.final_conv_size = last_nr_conv_filters * 7 * 7
            else:
                last_nr_conv_filters = 64 # default
                self.final_conv_size = 32 * 7 * 7
                self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)), # (20, 20)
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), # (9, 9)
                nn.ReLU(),
                init_(nn.Conv2d(64, last_nr_conv_filters, 3, stride=1)), # (7,7) --> taken and flattened!
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        else:
            last_nr_conv_filters = 64
            if self.input_110_84:
                self.final_conv_shape = (64, 3, 1)
                self.final_conv_size = 64 * 3 * 1
            elif self.less_dense:
                last_nr_conv_filters = 30
                self.final_conv_shape = (last_nr_conv_filters, 9, 6)
                self.final_conv_size = last_nr_conv_filters * 9 * 6
            else:
                self.final_conv_size = 64 * 9 * 6
                self.final_conv_shape = (64, 9, 6)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),  # (51, 39)
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), # (24, 18)
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, 4, stride=2)),  # (11, 8)
                nn.ReLU(),
                init_(nn.Conv2d(128, last_nr_conv_filters, 3, stride=1)), # (9, 6) --> taken and flattend
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        # count_parameters(self.main)
        self.train()

    @property
    def local_layer_depth(self):
        if self.downsample:
            return self.main[2].out_channels    
        else:
            return self.main[4].out_channels


    def forward(self, inputs, fmaps=False):
        if self.downsample:
            f5 = self.main[:4](inputs)
            f7 = self.main[4:6](f5)
            out = self.main[6:](f7)
        else:
            f5 = self.main[:6](inputs)
            f7 = self.main[6:8](f5)
            out = self.main[8:](f7)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'f7_not_permuted': f7,
                'f7': f7.permute(0, 2, 3, 1),
                'out': out
            }
        return out



class PPOEncoder(nn.Module):
    def __init__(self, env_name, checkpoint_index):
        super().__init__()
        checkpoint_step = checkpointed_steps_full_sorted[checkpoint_index]
        filepath = download_run(env_name, checkpoint_step)
        while not os.path.exists(filepath):
            time.sleep(5)

        self.masks = torch.zeros(1, 1)
        self.ppo_model, ob_rms = torch.load(filepath, map_location=lambda storage, loc: storage)

    def forward(self, x):
        _, _, _, _, feature_vectors, _ = self.ppo_model.act(x,
                                                            None,
                                                            self.masks,
                                                            deterministic=False)
        return feature_vectors
