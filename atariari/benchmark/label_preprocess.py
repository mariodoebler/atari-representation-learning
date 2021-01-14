from itertools import chain
import torch
import wandb
from benchmarking.utils.process_velocities import scalings, scalings_offsets


def remove_duplicates(tr_eps, val_eps, test_eps, test_labels):
    """
    Remove any items in test_eps (&test_labels) which are present in tr/val_eps
    """
    flat_tr = list(chain.from_iterable(tr_eps))
    flat_val = list(chain.from_iterable(val_eps))
    tr_val_set = set([x.numpy().tostring() for x in flat_tr] + [x.numpy().tostring() for x in flat_val])
    flat_test = list(chain.from_iterable(test_eps))

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
            print("Deleting {} for being too low in entropy! Sorry, dood!".format(k))
            low_entropy_labels.append(k)

    # just necessary to remove specific labels if they're needed for probing
    # for pretraining not necessary
    # probing: "invalid" {obs} episodes are marked as invalid by not attaching any labels
    # for pretraining del obs[key] would fail
    if train_mode == 'probe':
        for e in episode_labels:
            for obs in e:
                for key in low_entropy_labels:
                    del obs[key]
        # wandb.log(entropy_dict)
    return episode_labels, entropy_dict

# min_val = 100
# max_val = 0
# min_k = ""
# max_k = ""
def adjustLabelRange(labels, env_name, no_offsets=False):
    game_name = env_name.split('NoFrameskip-v4')[0]
    if no_offsets:
        scaling_factor = scalings[game_name]
    else:
        scaling_factor = scalings_offsets[game_name]
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

    
