import sys

from itertools import chain

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from benchmark.utils.process_velocities import scalings
from reinforcement_learning.utils.atari_offset_dict import getOffsetDict


def remove_duplicates(tr_eps, val_eps, test_eps, test_labels):
    """
    Remove any items in test_eps (&test_labels) which are present in tr/val_eps
    """
    flat_tr = list(chain.from_iterable(tr_eps))
    flat_val = list(chain.from_iterable(val_eps))
    tr_val_set = set([x.numpy().tostring() for x in flat_tr] + [x.numpy().tostring() for x in flat_val])
    flat_test = list(chain.from_iterable(test_eps))
    print("Start Duplicate Removal in Test-Set")

    for i, episode in enumerate(test_eps[:]):
        test_labels[i] = [label for obs, label in zip(test_eps[i], test_labels[i]) if obs.numpy().tostring() not in tr_val_set]
        test_eps[i] = [obs for obs in episode if obs.numpy().tostring() not in tr_val_set]
    test_len = len(list(chain.from_iterable(test_eps)))
    dups = len(flat_test) - test_len
    print('Duplicates: {}, Test Len: {}'.format(dups, test_len))
    #wandb.log({'Duplicates': dups, 'Test Len': test_len})
    return test_eps, test_labels


def remove_low_entropy_labels(episode_labels, entropy_threshold=0.3, train_mode="train_encoder"):
    flat_label_list = list(chain.from_iterable(episode_labels))
    counts = {}

    for label_dict in flat_label_list:
        for k in label_dict:
            counts[k] = counts.get(k, {})
            v = label_dict[k]
            counts[k][v] = counts[k].get(v, 0) + 1
    low_entropy_labels = []

    entropy_dict = {}
    for k in counts:
        entropy = torch.distributions.Categorical(
            torch.tensor([x / len(flat_label_list) for x in counts[k].values()])).entropy()
        entropy_dict['entropy_' + k] = entropy
        if entropy < entropy_threshold:
            print("Deleting {} for being too low in entropy! Sorry, dood! {:2f}".format(k, entropy.item()))
            low_entropy_labels.append(k)

    # just necessary to remove specific labels if they're needed for probing
    # for pretraining not necessary
    # probing: "invalid" {obs} episodes are marked as invalid by not attaching any labels
    # for pretraining del obs[key] would fail
    if train_mode == 'probe':
        for e in episode_labels:
            for obs in e:
                for key in low_entropy_labels:
                    if key in obs:
                        del obs[key]
        # wandb.log(entropy_dict)
    elif "train_encoder":
        pass
    else:
        sys.exit(f"Train mode {train_mode} doesn't exist, abort!")
    print("KEPT labels:")
    for l in episode_labels[0][0].keys():
        print(f"{l}")
    return episode_labels, entropy_dict

# min_val = 100
# max_val = 0
# min_k = ""
# max_k = ""
def scaleLabels(labels, env_name):
    game_name = env_name.split('NoFrameskip-v4')[0]
    scaling_factor = scalings[game_name]
    for l in labels:
        for i in l:
            for k, val in i.items():
                if ("_v_x" in k) or ("_v_y" in k):
                    # global min_val, min_k, max_val, max_k
                    # if val < min_val:
                    #     min_val = val  
                    #     min_k = k
                    # if val > max_val:
                    #     max_val = val  
                    #     max_k = k
                    # print(f"key is {k}")
                    i[k] = int(val * scaling_factor) + 128
                    # global list_velocities
                    # list_velocities.append(i[k])
                assert i[k] >= 0, f"is {i[k]}, val is {val} for key {k}"
                assert i[k] < 256, f"is {i[k]}, val is {val} for key {k}"
    # print(f"min val {min_val} for key {min_k}")
    # print(f"max val {max_val} for key {max_k}")
    # velocities = np.asarray(list_velocities)
    # np.save(os.path.join(Path.home(), "datadump", "velocities.npy"))
    return labels

def subtractOffsetsLabels(labels, env_name):
    game_name = env_name.split('NoFrameskip-v4')[0]
    offsets = getOffsetDict(game_name)
    for i, label_one_episode in enumerate(labels):
        for j, label in enumerate(label_one_episode):
            for k, value_of_key_of_label in label.items():
                # NO offsets for velocities as they're relative to the positions!!!
                # if ("_v_x" in k) or ("_v_y" in k):
                #     key_to_look_for = "_".join(k.split("_")[::2])
                #     offset_for_specific_key = self.offset[key_to_look_for]
                if ("_v_x" in k) or ("_v_y" in k):
                    offset_for_specific_key = 0
                elif ("_x" in k) or ("_y" in k):
                    offset_for_specific_key = offsets[k]
                else:
                    offset_for_specific_key = 0
                
                offset_corrected_value = value_of_key_of_label - offset_for_specific_key
                if "_x" in k and "_v_x" not in k:  # just x POSITION
                    offset_corrected_value = np.clip(offset_corrected_value, 0, 160)
                elif "_y" in k and "_v_y" not in k:  # just y POSITION
                    offset_corrected_value = np.clip(offset_corrected_value, 0, 210)
                labels[i][j][k] = offset_corrected_value

    return labels

