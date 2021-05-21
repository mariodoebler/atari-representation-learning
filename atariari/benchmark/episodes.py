import os
import time

from itertools import chain
from collections import deque

import torch
import numpy as np
import pickle5 as pickle

from PIL import Image
from torchvision import transforms

from .envs import make_vec_envs
from .utils import download_run
from .label_preprocess import (remove_duplicates, remove_low_entropy_labels,
                               scaleLabels, subtractOffsetsLabels)

from benchmarking.utils.helpers import (analyzeDebugEpisodes,
                                        countAndReportSampleNumbers,
                                        remove_invalid_episodes)
from benchmarking.utils.process_dataset import convertDataType

try:
    import wandb
except:
    pass

from PIL import Image

checkpointed_steps_full = [10753536, 1076736, 11828736, 12903936, 13979136, 15054336, 1536, 16129536, 17204736,
                           18279936,
                           19355136, 20430336, 21505536, 2151936, 22580736, 23655936, 24731136, 25806336, 26881536,
                           27956736,
                           29031936, 30107136, 31182336, 32257536, 3227136, 33332736, 34407936, 35483136, 36558336,
                           37633536,
                           38708736, 39783936, 40859136, 41934336, 43009536, 4302336, 44084736, 45159936, 46235136,
                           47310336,
                           48385536, 49460736, 49999872, 5377536, 6452736, 7527936, 8603136, 9678336]

checkpointed_steps_full_sorted = [1536, 1076736, 2151936, 3227136, 4302336, 5377536, 6452736, 7527936, 8603136, 9678336,
                                  10753536, 11828736, 12903936, 13979136, 15054336, 16129536, 17204736, 18279936,
                                  19355136, 20430336, 21505536, 22580736, 23655936, 24731136, 25806336, 26881536,
                                  27956736, 29031936, 30107136, 31182336, 32257536, 33332736, 34407936, 35483136,
                                  36558336, 37633536, 38708736, 39783936, 40859136, 41934336, 43009536, 44084736,
                                  45159936, 46235136, 47310336, 48385536, 49460736, 49999872]


def get_random_agent_rollouts(env_name, steps, seed=42, num_processes=1, num_frame_stack=1, downsample=False, color=False, use_extended_wrapper=False, train_mode="train_encoder", wandb=None):
    envs = make_vec_envs(env_name, seed, num_processes, num_frame_stack, downsample,
                         color, use_extended_wrapper=use_extended_wrapper, train_mode=train_mode)
    envs.reset()
    episode_rewards = deque(maxlen=10)
    print('-------Collecting samples----------')
    debug_save_frames_for_plotting = True and not torch.cuda.is_available()
    # just works if num_processes = 1!!!
    if debug_save_frames_for_plotting and num_processes > 1:
        num_processes = 1
        print("Set Num Processes to 1 as otherwise no proper dumping of image-observations possible")
    # (n_processes * n_episodes * episode_len)
    episodes = [[[]] for _ in range(num_processes)]
    episode_labels = [[[]] for _ in range(num_processes)]
    # do NOT do this on the server, just for testing on computer
    if debug_save_frames_for_plotting:
        print(f"frame-set\t\tvel1\t\tvel2")
    for step in range(steps // num_processes):
        # Take action using a random policy
        action = torch.tensor(
            np.array([np.random.randint(1, envs.action_space.n) for _ in range(num_processes)])) \
            .unsqueeze(dim=1)
        # obs.shape: [NUM_PROCESSES, num-stacks, height, width]
        obs, reward, done, infos = envs.step(action)
        if debug_save_frames_for_plotting and step < 200 and os.path.exists('/home/cathrin/MA/datadump/'):
            # img_obs = envs.render('rgb_array')
            # im = Image.fromarray(img_obs)
            im_obs = obs.squeeze()
            if num_frame_stack == 4:
                im = im_obs[0, :, :]
                for i in range(1, 4):
                    # 3, frameestack, height, width
                    im = torch.cat((im, im_obs[i, :, :]), axis=-1)  # via the last axis --> width
            elif num_frame_stack == 1:
                im = im_obs
            im = transforms.ToPILImage()(im).convert("RGB")
            im.save(f'/home/cathrin/MA/datadump/trash/{step}.png')
            # im.save(f'/home/cathrin/MA/datadump/img_obs_{step}.png')
            torch.save(
                obs, f"/home/cathrin/MA/datadump/observations/obs_{step}.pt")
            if use_extended_wrapper:
                if 'pong' in env_name.lower() and infos[0].get('labels', None):
                    print(
                        f"{step}\t\t\t{infos[0]['labels']['ball_v_x']}\t\t\t{infos[0]['labels']['ball_v_y']}")
                elif 'pacman' in env_name.lower() and infos[0].get('labels', None):
                    print(
                        f"{step}\t\t\t{infos[0]['labels']['player_v_x']}\t\t\t{infos[0]['labels']['enemy_sue_v_x']}")
                elif 'breakout' in env_name.lower() and infos[0].get('labels', None):
                    print(
                        f"{step}\t\t\t{infos[0]['labels']['player_v_x']}\t\t\t{infos[0]['labels']['ball_v_y']}")
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            if done[i] != 1:
                episodes[i][-1].append(obs[i].clone())
                if "labels" in info.keys():
                    episode_labels[i][-1].append(info["labels"])
            else:
                episodes[i].append([obs[i].clone()])
                if "labels" in info.keys():
                    episode_labels[i].append([info["labels"]])

    # Convert to 2d list from 3d list
    episodes = list(chain.from_iterable(episodes))
    # Convert to 2d list from 3d list
    episode_labels = list(chain.from_iterable(episode_labels))
    envs.close()
    mean_episode_reward = np.mean(episode_rewards)
    try:
        wandb.log({'mean_reward': mean_episode_reward})
    except:
        pass
    return episodes, episode_labels

def get_filepath_dataset(dataset_path, env_name):
    files = os.listdir(dataset_path)
    pkl_files = [f for f in files if f.endswith("_processed.pkl")]
    filepath = [os.path.join(dataset_path, f) for f in pkl_files if env_name.lower() in f.lower()]
    assert len(filepath) == 1, f"ERROR: there is {len(filepath)} processed.pkl file for the game {env_name} in {dataset_path}!"
    print(f"Filepath for game {env_name} is: {filepath[0]}")
    return filepath[0]



def get_preprocessed_benchmark_dataset(env_name, steps):
    gpu = torch.cuda.is_available()
    if gpu:
        dataset_path = '/data/private/atari_datasets/processed_pretrained_datasets/'
    else:
        # cpu / prototyping laptop
        dataset_path = '/home/cathrin/MA/datadump/dev_dataset_benchmark/breakout/'

    print(f"Dataset path for the *processed.pkl is {dataset_path}")

    filepath_dataset = get_filepath_dataset(dataset_path, env_name)
    with open(filepath_dataset, 'rb') as f:
        data = pickle.load(f)
    # except ValueError:  # pickle5...
    #     import pickle5 as pickle

    #     with open(filepath_dataset, 'rb') as f:
    #         data = pickle.load(f)
    print(f"loaded data via pickle")
    tr_labels, val_labels, test_labels = data["training_labels"], data["validation_labels"], data["test_labels"]
    verify_amount_steps(tr_labels, val_labels, test_labels, steps_wanted=steps, debugging=not gpu)
    tr_eps, val_eps, test_eps = data["training_episodes"], data["validation_episodes"], data["test_episodes"]
    tr_eps = convertDataType(tr_eps, torch.float32)
    val_eps = convertDataType(val_eps, torch.float32)
    test_eps = convertDataType(test_eps, torch.float32)
    return tr_eps, val_eps, test_eps, tr_labels, val_labels, test_labels

def verify_amount_steps(tr, val, test, steps_wanted, debugging):
    amount_steps_tr = [len(e) for e in tr]
    amount_steps_val = [len(e) for e in val]
    amount_steps_test = [len(e) for e in test]

    total_amount_steps = sum(amount_steps_tr) + sum(amount_steps_val) + sum(amount_steps_test)
    print(f"Required {steps_wanted}, found {total_amount_steps}")
    if not debugging:
        assert total_amount_steps >= steps_wanted, f"Required {steps_wanted} but just {total_amount_steps} loaded!"
        assert total_amount_steps < steps_wanted + 3000  # not possible to get exactly 50k...
    

def get_ppo_rollouts(env_name, steps, seed=42, num_processes=1,
                     num_frame_stack=1, downsample=False, color=False, checkpoint_index=-1, use_extended_wrapper=False, just_use_one_input_dim=True, train_mode="train_encoder"):
    checkpoint_step = checkpointed_steps_full_sorted[checkpoint_index]
    filepath = download_run(env_name, checkpoint_step)
    while not os.path.exists(filepath):
        time.sleep(5)

    envs = make_vec_envs(env_name, seed,  num_processes, num_frame_stack, downsample,
                         color, use_extended_wrapper=use_extended_wrapper, train_mode=train_mode)

    # filepath =
    actor_critic, ob_rms = torch.load(
        filepath, map_location=lambda storage, loc: storage)

    # (n_processes * n_episodes * episode_len)
    episodes = [[[]] for _ in range(num_processes)]
    episode_labels = [[[]] for _ in range(num_processes)]
    episode_rewards = deque(maxlen=10)

    masks = torch.zeros(1, 1)
    obs = envs.reset()
    if just_use_one_input_dim:
        obs = obs[:, -1, :, :]
        obs = obs.unsqueeze(1)
    entropies = []
    for step in range(steps // num_processes):
        # Take action using the PPO policy
        with torch.no_grad():
            _, action, _, _, actor_features, dist_entropy = actor_critic.act(
                obs, None, masks, deterministic=False)
        action = torch.tensor([envs.action_space.sample() if np.random.uniform(0, 1) < 0.2 else action[i]
                               for i in range(num_processes)]).unsqueeze(dim=1)
        entropies.append(dist_entropy.clone())
        obs, reward, done, infos = envs.step(action)
        if just_use_one_input_dim:
            obs = obs[:, -1, :, :]
            obs = obs.unsqueeze(1)
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            if done[i] != 1:
                episodes[i][-1].append(obs[i].clone())
                if "labels" in info.keys():
                    episode_labels[i][-1].append(info["labels"])
            else:
                episodes[i].append([obs[i].clone()])
                if "labels" in info.keys():
                    episode_labels[i].append([info["labels"]])

    # Convert to 2d list from 3d list
    episodes = list(chain.from_iterable(episodes))
    # Convert to 2d list from 3d list
    episode_labels = list(chain.from_iterable(episode_labels))
    mean_entropy = torch.stack(entropies).mean()
    mean_episode_reward = np.mean(episode_rewards)
    try:
        wandb.log({'action_entropy': mean_entropy,
                   'mean_reward': mean_episode_reward})
    except:
        pass

    return episodes, episode_labels


def get_episodes(env_name,
                 steps,
                 seed=42,
                 num_processes=1,
                 num_frame_stack=1,
                 downsample=False,
                 color=False,
                 entropy_threshold=0.6,
                 collect_mode="random_agent",
                 train_mode="probe",
                 checkpoint_index=-1,
                 min_episode_length=64,
                 wandb=None,
                 use_extended_wrapper=False,
                 just_use_one_input_dim=True,
                 no_offsets=False,
                 collect_for_curl=False): # curl: do not split into train/val

    if collect_mode == "random_agent":
        # List of episodes. Each episode is a list of 160x210 observations
        episodes, episode_labels = get_random_agent_rollouts(env_name=env_name,
                                                             steps=steps,
                                                             seed=seed,
                                                             num_processes=num_processes,
                                                             num_frame_stack=num_frame_stack,
                                                             downsample=downsample, color=color,
                                                             use_extended_wrapper=use_extended_wrapper,
                                                            #  no_offsets=no_offsets,
                                                             train_mode=train_mode, wandb=wandb)

    elif collect_mode == "preprocessed_benchmark_dataset":
        tr_eps, val_eps, test_eps, tr_labels, val_labels, test_labels = get_preprocessed_benchmark_dataset(env_name=env_name,
                                                             steps=steps)
                                                            #  seed=seed,
                                                            #  num_processes=num_processes,
                                                            #  num_frame_stack=num_frame_stack,
                                                            #  downsample=downsample, color=color,
                                                            #  use_extended_wrapper=use_extended_wrapper,
                                                            #  no_offsets=no_offsets,
                                                            #  train_mode=train_mode)

    elif collect_mode == "pretrained_ppo":

        # List of episodes. Each episode is a list of 160x210 observations
        episodes, episode_labels = get_ppo_rollouts(env_name=env_name,
                                                  steps=steps,
                                                  seed=seed,
                                                  num_processes=num_processes,
                                                  num_frame_stack=num_frame_stack,
                                                  downsample=downsample,
                                                  color=color,
                                                  checkpoint_index=checkpoint_index, use_extended_wrapper=use_extended_wrapper,
                                                  just_use_one_input_dim=just_use_one_input_dim,
                                                #   no_offsets=no_offsets,
                                                  train_mode=train_mode)


    else:
        assert False, "Collect mode {} not recognized".format(collect_mode)

    if collect_mode != "preprocessed_benchmark_dataset":
        ep_inds = [i for i in range(len(episodes)) if len(episodes[i]) > min_episode_length]
        episodes = [episodes[i] for i in ep_inds]
        print(f"len episode labels {len(episode_labels)}, ep_inds are {*ep_inds,}")
        print(f"len episodes: {len(episodes)} min length: {min_episode_length}")
        if train_mode == "probe":
            episodes, episode_labels = remove_invalid_episodes(episodes, episode_labels, frame_stack=num_frame_stack, wandb=wandb)
            episode_labels = [episode_labels[i] for i in ep_inds]
        # if num_frame_stack == 4:
        #     analyzeDebugEpisodes(episodes, batch_size=min_episode_length, env_name=env_name.lower())
        #     sys.exit(0)  # successfull termination
            episode_labels, entropy_dict = remove_low_entropy_labels(episode_labels, entropy_threshold=entropy_threshold, train_mode=train_mode)

        try:
            wandb.log(entropy_dict)
        except:
            pass

        inds = np.arange(len(episodes))
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(inds)
        print(f"inds shuffled are {*inds,}")

    # important: FIRST subtract offsets, then SCALE! 
    if not no_offsets and train_mode == "probe":
        if "battle" not in env_name.lower() and "space" not in env_name.lower() and "frost" not in env_name.lower():
            if collect_mode == "preprocessed_benchmark_dataset":
                tr_labels = subtractOffsetsLabels(tr_labels, env_name)
                val_labels = subtractOffsetsLabels(val_labels, env_name)
                test_labels = subtractOffsetsLabels(test_labels, env_name)
            else:
                episode_labels = subtractOffsetsLabels(episode_labels, env_name)

    if (use_extended_wrapper and train_mode == "probe"): 
        # scaling depends whether offsets have been subtracted or not!
        if collect_mode == "preprocessed_benchmark_dataset":
            tr_labels = scaleLabels(tr_labels, env_name)
            val_labels = scaleLabels(val_labels, env_name)
            test_labels = scaleLabels(test_labels, env_name)
        else:
            episode_labels = scaleLabels(episode_labels, env_name)


    if train_mode == "train_encoder":
        if collect_for_curl:
            return episodes
        assert len(inds) > 1, "Not enough episodes to split into train and val. You must specify enough steps to get at least two episodes"
        split_ind = int(0.8 * len(inds))
        tr_eps, val_eps = episodes[:split_ind], episodes[split_ind:]
        countAndReportSampleNumbers(training=tr_eps, validation=val_eps)
        return tr_eps, val_eps

    if train_mode == "probe":
        if not collect_mode == "preprocessed_benchmark_dataset":
            val_split_ind, te_split_ind = int(0.7 * len(inds)), int(0.8 * len(inds))
            assert val_split_ind > 0 and te_split_ind > val_split_ind,\
                "Not enough episodes to split into train, val and test. You must specify more steps"
            tr_eps, val_eps, test_eps = episodes[:val_split_ind], episodes[val_split_ind:te_split_ind], episodes[
                te_split_ind:]
            tr_labels, val_labels, test_labels = episode_labels[:val_split_ind], \
            episode_labels[val_split_ind:te_split_ind], episode_labels[te_split_ind:]
            test_eps, test_labels = remove_duplicates(tr_eps, val_eps, test_eps, test_labels)
            test_ep_inds = [i for i in range(len(test_eps)) if len(test_eps[i]) > 1]
            test_eps = [test_eps[i] for i in test_ep_inds]
            test_labels = [test_labels[i] for i in test_ep_inds]
            countAndReportSampleNumbers(training=tr_labels, validation=val_labels, test=test_labels, wandb=wandb)
        return tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels

    if train_mode == "dry_run":
        return episodes, episode_labels
