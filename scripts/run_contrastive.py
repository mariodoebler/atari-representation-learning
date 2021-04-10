import time

from itertools import chain
from collections import deque

from atariari.methods.cpc import CPCTrainer
from atariari.methods.vae import VAETrainer
from atariari.methods.stdim import InfoNCESpatioTemporalTrainer
from atariari.methods.utils import get_argparser
from atariari.methods.encoders import ImpalaCNN, NatureCNN
from atariari.methods.jsd_stdim import SpatioTemporalTrainer
from atariari.methods.dim_baseline import DIMTrainer
from atariari.methods.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from atariari.methods.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from atariari.methods.no_action_feedforward_predictor import NaFFPredictorTrainer
from atariari.benchmark.episodes import get_episodes

import torch
import wandb
import numpy as np


def train_encoder(args, wandb):
    device = torch.device("cuda:" + str(args.cuda_id)
                          if torch.cuda.is_available() else "cpu")
    tr_eps, val_eps = get_episodes(steps=args.pretraining_steps,
                                   env_name=args.env_name,
                                   seed=args.seed,
                                   num_processes=args.num_processes,
                                   num_frame_stack=args.num_frame_stack,
                                   downsample=not args.no_downsample,
                                   color=args.color,
                                   entropy_threshold=args.entropy_threshold,
                                   collect_mode=args.probe_collect_mode,
                                   train_mode="train_encoder",
                                   checkpoint_index=args.checkpoint_index,
                                   min_episode_length=args.batch_size,
                                   wandb=wandb,
                                   use_extended_wrapper=args.use_extended_wrapper)

    observation_shape = tr_eps[0][0].shape
    if args.encoder_type == "Nature":
        args.less_dense = False
        args.more_dense = False
        args.more_spatial_dim = False
        encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)
    encoder.to(device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack
    if args.method == 'cpc':
        trainer = CPCTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'jsd-stdim':
        trainer = SpatioTemporalTrainer(
            encoder, config, device=device, wandb=wandb)
    elif args.method == 'vae':
        trainer = VAETrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "naff":
        trainer = NaFFPredictorTrainer(
            encoder, config, device=device, wandb=wandb)
    elif args.method == "infonce-stdim":
        trainer = InfoNCESpatioTemporalTrainer(
            encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(
            encoder, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(
            encoder, config, device=device, wandb=wandb)
    elif args.method == "dim":
        trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    trainer.train(tr_eps, val_eps, passing_file=args.passing_file)

    return encoder


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tags = [device.type, 'pretraining-only', "fs: " + str(args.num_frame_stack), args.env_name, args.encoder_type, "batch size: " + str(
        args.batch_size), "pretraining-steps: " + str(args.pretraining_steps), "epochs: " + str(args.epochs)]
    # kind of bad programming, but needed as all further previous code builds up on 'NO_downsample'...
    args.no_downsample = not args.downsample
    if args.downsample:
        tags.append("downsample84")
    config = {}
    config.update(vars(args))
    if args.wandb_off:
        wandb = None
    else:
        if not args.name_logging:
            wandb.init(project=args.wandb_proj, tags=tags)
        else:
            wandb.init(project=args.wandb_proj,
                       tags=tags, name=args.name_logging)
        wandb.config.update(config)
    train_encoder(args, wandb)
