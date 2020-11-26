import os
import sys

import torch
import wandb

from scripts.run_contrastive import train_encoder

from atariari.methods.utils import (get_argparser, probe_only_methods,
                                    train_encoder_methods)
from atariari.methods.encoders import ImpalaCNN, NatureCNN, PPOEncoder
from atariari.methods.majority import majority_baseline
from atariari.benchmark.probe import ProbeTrainer
from atariari.benchmark.episodes import get_episodes


def run_probe(args):
    wandb.config.update(vars(args))
    tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels = get_episodes(steps=args.probe_steps,
                                                                                 env_name=args.env_name,
                                                                                 seed=args.seed,
                                                                                 num_processes=args.num_processes,
                                                                                 num_frame_stack=args.num_frame_stack,
                                                                                 downsample=not args.no_downsample,
                                                                                 color=args.color,
                                                                                 entropy_threshold=args.entropy_threshold,
                                                                                 collect_mode=args.probe_collect_mode,
                                                                                 train_mode="probe",
                                                                                 checkpoint_index=args.checkpoint_index,
                                                                                 min_episode_length=args.batch_size)

    print("got episodes!")

    if args.train_encoder and args.method in train_encoder_methods:
        print("Training encoder from scratch")
        encoder = train_encoder(args)
        encoder.probing = True
        encoder.eval()

    elif args.method == "pretrained-rl-agent":
        encoder = PPOEncoder(args.env_name, args.checkpoint_index)

    elif args.method == "majority":
        encoder = None

    else:
        observation_shape = tr_eps[0][0].shape
        if args.encoder_type == "Nature":
            encoder = NatureCNN(observation_shape[0], args)
        elif args.encoder_type == "Impala":
            encoder = ImpalaCNN(observation_shape[0], args)

        if args.weights_path == "None":
            if args.method not in probe_only_methods:
                sys.stderr.write("Probing without loading in encoder weights! Are sure you want to do that??")
        else:
            print("Print loading in encoder weights from probe of type {} from the following path: {}"
                  .format(args.method, args.weights_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            encoder.load_state_dict(torch.load(args.weights_path, map_location=device))
            encoder.eval()

    torch.set_num_threads(1)

    if args.method == 'majority':
        test_acc, test_f1score = majority_baseline(tr_labels, test_labels, wandb)

    else:
        trainer = ProbeTrainer(encoder=encoder,
                               epochs=args.epochs,
                               method_name=args.method,
                               lr=args.probe_lr,
                               batch_size=args.batch_size,
                               patience=args.patience,
                               wandb=wandb,
                               fully_supervised=(args.method == "supervised"),
                               save_dir=wandb.run.dir)

        trainer.train(tr_eps, val_eps, tr_labels, val_labels)
        test_acc, test_f1score = trainer.test(test_eps, test_labels)
        # trainer = SKLearnProbeTrainer(encoder=encoder)
        # test_acc, test_f1score = trainer.train_test(tr_eps, val_eps, tr_labels, val_labels,
        #                                             test_eps, test_labels)

    print(test_acc, test_f1score)
    wandb.log(test_acc)
    wandb.log(test_f1score)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    if (args.weights_path and args.passing_file) is None:
        args.train_encoder = False

    # if args.batch_size > args.num_processes:
    #     print(f"Batch size was set to {args.batch_size} but should be maximum {args.num_processes} (args.num-processes)")
    #     sys.exit(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tags = [device.type, 'probe', "fs: " + str(args.num_frame_stack) , args.env_name, args.encoder_type, "batch size: " + str(args.batch_size), "pretraining-steps: " + str(args.pretraining_steps), "probe steps: " + str(args.probe_steps), "epochs: " + str(args.epochs)]
    if args.wandb_off:
        os.environ["WANDB_MODE"] ="dryrun"
    wandb.init(project=args.wandb_proj, entity=args.wandb_entity, tags=tags)
    #print(f"Running now for environment {args.env-name}")
    run_probe(args)
