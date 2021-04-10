from copy import deepcopy
from typing import List, Union

import numpy as np

import torch

from .utils import (append_suffix, appendabledict,
                    calculate_multiclass_accuracy,
                    calculate_multiclass_f1_score, compute_dict_average,
                    EarlyStopping)
from .categorization import regression_keys, summary_key_dict

from torch import nn
from torch.utils.data import BatchSampler, RandomSampler

from benchmarking.utils.helpers import (calculate_mae_regression_score,
                                        calculate_top11_regression_score,
                                        combineMetricsPerCategory,
                                        createTableList)
from benchmarking.utils.categorization_extended import (regression_keys_extended,
                                                        summary_key_dict_extended)


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.model = nn.Linear(in_features=input_dim, out_features=num_classes)

    def forward(self, feature_vectors):
        return self.model(feature_vectors)


class FullySupervisedLinearProbe(nn.Module):
    def __init__(self, encoder, num_classes=255):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.probe = LinearProbe(input_dim=self.encoder.hidden_size,
                                 num_classes=num_classes)

    def forward(self, x):
        feature_vec = self.encoder(x)
        return self.probe(feature_vec)


class ProbeTrainer():
    def __init__(self,
                 encoder=None,
                 method_name="my_method",
                 wandb=None,
                 patience=15,
                 num_classes=256,
                 fully_supervised=False,
                 save_dir=".models",
                 device=torch.device(
                     "cuda" if torch.cuda.is_available() else "cpu"),
                 lr=5e-4,
                 epochs=100,
                 batch_size=64,
                 representation_len=256,
                 just_use_one_input_dim=False,
                 use_extended_wrapper=True):

        self.encoder = encoder
        self.wandb = wandb
        self.device = device
        self.fully_supervised = fully_supervised
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.method = method_name
        self.feature_size = representation_len
        self.loss_fn = nn.CrossEntropyLoss()
        self.just_use_one_input_dim = just_use_one_input_dim
        self.use_extended_wrapper = use_extended_wrapper

        # bad convention, but these get set in "create_probes"
        self.probes = self.early_stoppers = self.optimizers = self.schedulers = None

    def create_probes(self, sample_label):
        if self.fully_supervised:
            assert self.encoder != None, "for fully supervised you must provide an encoder!"
            self.probes = {k: FullySupervisedLinearProbe(encoder=self.encoder,
                                                         num_classes=self.num_classes).to(self.device) for k in
                           sample_label.keys()}
        else:
            self.probes = {k: LinearProbe(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

        self.early_stoppers = {
            k: EarlyStopping(patience=self.patience, verbose=False,
                             name=k + "_probe", save_dir=self.save_dir)
            for k in sample_label.keys()}

        self.optimizers = {k: torch.optim.Adam(list(self.probes[k].parameters()),
                                               eps=1e-5, lr=self.lr) for k in sample_label.keys()}
        self.schedulers = {
            k: torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizers[k], patience=5, factor=0.2, verbose=True,
                                                          mode='max', min_lr=1e-5) for k in sample_label.keys()}

        print("The deployed labels are:")
        print([f"{l}" for l in sample_label.keys()])

    def generate_batch(self, episodes, episode_labels):
        total_steps = sum([len(e) for e in episodes])
        assert total_steps > self.batch_size
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)

        counter = 0
        for indices in sampler:
            counter += 1
            episodes_batch = [episodes[x] for x in indices]
            episode_labels_batch = [episode_labels[x] for x in indices]
            xs, labels = [], appendabledict()
            for ep_ind, episode in enumerate(episodes_batch):
                # Get one sample from this episode
                t = np.random.randint(len(episode))
                x = episode[t]
                if self.just_use_one_input_dim:
                    x = torch.unsqueeze(x[-1, :, :], 0)
                xs.append(x)
                labels.append_update(episode_labels_batch[ep_ind][t])
            yield torch.stack(xs).float().to(self.device) / 255., labels
        print(f"yielded {counter} batches")

    def probe(self, batch, k):
        probe = self.probes[k]
        probe.to(self.device)
        if self.fully_supervised:
            # if method is supervised batch is a batch of frames and probe is a full encoder + linear or nonlinear probe
            preds = probe(batch)

        elif not self.encoder:
            # if encoder is None then inputs are vectors
            f = batch.detach()
            assert len(f.squeeze(
            ).shape) == 2, "if input is not a batch of vectors you must specify an encoder!"
            preds = probe(f)

        else:
            with torch.no_grad():
                self.encoder.to(self.device)
                f = self.encoder(batch).detach()  # RANGE [0, 1]!!!
            if len(f.shape) == 4:
                f = f.reshape(f.size(0), -1)
            preds = probe(f)
        return preds

    def do_one_epoch(self, episodes, label_dicts):
        # min_max = [100, 0]
        # min_key, max_key = None, None
        sample_label = label_dicts[0][0]
        epoch_loss, accuracy = {k + "_loss": [] for k in sample_label.keys() if
                                not self.early_stoppers[k].early_stop}, \
                               {k + "_acc": [] for k in sample_label.keys() if
                                not self.early_stoppers[k].early_stop}

        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                if self.early_stoppers[k].early_stop:
                    continue
                optim = self.optimizers[k]
                optim.zero_grad()

                label = torch.tensor(label).long().to(self.device)
                preds = self.probe(x, k)  # preds.shape [B, 256]

                # if torch.min(label) < min_max[0]:
                #     min_max[0] = torch.min(label)
                #     min_key = k
                # if torch.max(label) > min_max[1]:
                #     min_max[1] = torch.max(label)
                #     max_key = k
                loss = self.loss_fn(preds, label)

                epoch_loss[k + "_loss"].append(loss.detach().item())
                preds = preds.cpu().detach().numpy()
                preds = np.argmax(preds, axis=1)
                label = label.cpu().detach().numpy()
                accuracy[k + "_acc"].append(calculate_multiclass_accuracy(preds,
                                                                          label))
                if self.probes[k].training:
                    loss.backward()
                    optim.step()

        epoch_loss = {k: np.mean(loss) for k, loss in epoch_loss.items()}
        accuracy = {k: np.mean(acc) for k, acc in accuracy.items()}

        return epoch_loss, accuracy

    def do_test_epoch(self, episodes, label_dicts, regression_keys=[]):
        sample_label = label_dicts[0][0]
        accuracy_dict, f1_score_dict, mae_regression_score_dict, top11_regression_score_dict = {}, {}, {}, {}
        pred_dict, all_label_dict = {k: [] for k in sample_label.keys()}, \
                                    {k: [] for k in sample_label.keys()}

        # collect all predictions first
        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                label = torch.tensor(label).long().cpu()
                all_label_dict[k].append(label)
                preds = self.probe(x, k).detach().cpu()
                pred_dict[k].append(preds)

        for k in all_label_dict.keys():
            preds, labels = torch.cat(pred_dict[k]).cpu().detach().numpy(),\
                torch.cat(all_label_dict[k]).cpu().detach().numpy()

            preds = np.argmax(preds, axis=1)
            accuracy = calculate_multiclass_accuracy(preds, labels)
            f1score = calculate_multiclass_f1_score(preds, labels)
            accuracy_dict[k] = accuracy
            f1_score_dict[k] = f1score
            if k in regression_keys:
                mae_regression_score_dict[k] = calculate_mae_regression_score(
                    preds, labels, label_key=k)
                top11_regression_score_dict[k] = calculate_top11_regression_score(
                    preds, labels, label_key=k)

        return accuracy_dict, f1_score_dict, mae_regression_score_dict, top11_regression_score_dict

    def train(self, tr_eps, val_eps, tr_labels, val_labels):
        # if not self.encoder:
        #     assert len(tr_eps[0][0].squeeze().shape) == 2, "if input is a batch of vectors you must specify an encoder!"
        sample_label = tr_labels[0][0]
        self.create_probes(sample_label)
        e = 0
        all_probes_stopped = np.all(
            [early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        while (not all_probes_stopped) and e < self.epochs:
            epoch_loss, accuracy = self.do_one_epoch(tr_eps, tr_labels)
            self.log_results(e, epoch_loss, accuracy)

            val_loss, val_accuracy = self.evaluate(
                val_eps, val_labels, epoch=e)
            # update all early stoppers
            for k in sample_label.keys():
                if not self.early_stoppers[k].early_stop:
                    self.early_stoppers[k](
                        val_accuracy["val_" + k + "_acc"], self.probes[k])

            for k, scheduler in self.schedulers.items():
                if not self.early_stoppers[k].early_stop:
                    scheduler.step(val_accuracy['val_' + k + '_acc'])
            e += 1
            all_probes_stopped = np.all(
                [early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        print("All probes early stopped!")

    def evaluate(self, val_episodes, val_label_dicts, epoch=None):
        for k, probe in self.probes.items():
            probe.eval()
        epoch_loss, accuracy = self.do_one_epoch(val_episodes, val_label_dicts)
        epoch_loss = {"val_" + k: v for k, v in epoch_loss.items()}
        accuracy = {"val_" + k: v for k, v in accuracy.items()}
        self.log_results(epoch, epoch_loss, accuracy)
        for k, probe in self.probes.items():
            probe.train()
        return epoch_loss, accuracy

    def test(self, test_episodes, test_label_dicts, epoch=None):
        for k in self.early_stoppers.keys():
            self.early_stoppers[k].early_stop = False
        for k, probe in self.probes.items():
            probe.eval()

        if self.use_extended_wrapper:
            regression_keys_ext = regression_keys_extended
        else:
            regression_keys_ext = []
        regression_keys_list = list(set(regression_keys + regression_keys_ext))
        # regression_keys: ALL possible (from ALL games)
        acc_dict, f1_dict, mae_regression_dict, top11_regression_dict = self.do_test_epoch(
            test_episodes, test_label_dicts, regression_keys=regression_keys_list)

        # for regression-metrics-comparison
        wanted_categories = list(set(list(
            summary_key_dict_extended.keys()) + ["across_categories_avg", "overall_avg"]))
        acc_dict, f1_dict, mae_regression_dict, mae_f1_dict, top11_regression_dict, top11_f1_dict, metrics_per_category_dict, table_test = postprocess_raw_metrics(
            acc_dict, f1_dict, mae_regression_dict, top11_regression_dict, use_extended_wrapper=self.use_extended_wrapper, wanted_categories=wanted_categories, regression_keys_list=regression_keys_list)
        print("""In our paper, we report F1 scores and accuracies averaged across each category. 
              That is, we take a mean across all state variables in a category to get the average score for that category.
              Then we average all the category averages to get the final score that we report per game for each method. 
              These scores are called \'across_categories_avg_acc\' and \'across_categories_avg_f1\' respectively
              We do this to prevent categories with large number of state variables dominating the mean F1 score.
              """)

        acc_dict, f1_dict, mae_regression_dict, mae_f1_dict, top11_regression_dict, top11_f1_dict = self.roundResults(
            acc_dict=acc_dict, f1_dict=f1_dict, mae_regression_dict=mae_regression_dict, mae_f1_dict=mae_f1_dict, top11_regression_dict=top11_regression_dict, top11_f1_dict=top11_f1_dict)
        self.log_results("Test", acc_dict, f1_dict, mae_regression_dict,
                         mae_f1_dict, top11_regression_dict, top11_f1_dict)
        return acc_dict, f1_dict, mae_regression_dict, mae_f1_dict, top11_regression_dict, top11_f1_dict, metrics_per_category_dict, table_test

    def roundResults(self, **kwargs):
        metrics = []
        for metric in kwargs:
            metric = {k: np.round(v, 4) for k, v in kwargs[metric].items()}
            metrics.append(deepcopy(metric))
        return metrics

    def log_results(self, epoch_idx, *dictionaries):
        print("Epoch: {}".format(epoch_idx))
        for dictionary in dictionaries:
            for k, v in dictionary.items():
                print("\t {}: {:8.4f}".format(k, v))
            print("\t --")


def postprocess_raw_metrics(acc_dict, f1_dict, mae_regression_dict, top11_regression_dict, use_extended_wrapper, wanted_categories: List[str] = [], regression_keys_list: List[str] = []) -> Union[dict, dict, dict, dict, dict]:
    mae_f1_dict = combineRegressionAndF1Dicts(
        regression_dict=mae_regression_dict, f1_dict=f1_dict)
    top11_f1_dict = combineRegressionAndF1Dicts(
        regression_dict=top11_regression_dict, f1_dict=f1_dict)

    acc_overall_avg, f1_overall_avg, mae_regression_overall_avg, mae_f1_overall_avg, top11_regression_overall_avg, top11_f1_overall_avg = compute_dict_average(acc_dict), \
        compute_dict_average(f1_dict), \
        compute_dict_average(mae_regression_dict), \
        compute_dict_average(mae_f1_dict), \
        compute_dict_average(top11_regression_dict), \
        compute_dict_average(top11_f1_dict)

    acc_category_avgs_dict, f1_category_avgs_dict, mae_regression_category_avgs_dict, mae_f1_category_avgs_dict, top11_regression_category_avgs_dict, top11_f1_category_avgs_dict = \
        compute_category_avgs(acc_dict, use_extended_wrapper), \
        compute_category_avgs(f1_dict, use_extended_wrapper), \
        compute_category_avgs(mae_regression_dict, use_extended_wrapper), \
        compute_category_avgs(mae_f1_dict, use_extended_wrapper), \
        compute_category_avgs(top11_regression_dict, use_extended_wrapper), \
        compute_category_avgs(top11_f1_dict, use_extended_wrapper)

    acc_avg_across_categories, f1_avg_across_categories, mae_regression_avg_across_categories, mae_f1_avg_across_categories, top11_regression_avg_across_categories, top11_f1_avg_across_categories = \
        compute_dict_average(acc_category_avgs_dict), \
        compute_dict_average(f1_category_avgs_dict), \
        compute_dict_average(mae_regression_category_avgs_dict), \
        compute_dict_average(mae_f1_category_avgs_dict), \
        compute_dict_average(top11_regression_category_avgs_dict), \
        compute_dict_average(top11_f1_category_avgs_dict)

    acc_dict.update(acc_category_avgs_dict)
    f1_dict.update(f1_category_avgs_dict)
    mae_regression_dict.update(mae_regression_category_avgs_dict)
    mae_f1_dict.update(mae_f1_category_avgs_dict)
    top11_regression_dict.update(top11_regression_category_avgs_dict)
    top11_f1_dict.update(top11_f1_category_avgs_dict)

    acc_dict["overall_avg"], f1_dict["overall_avg"] = acc_overall_avg, f1_overall_avg
    mae_regression_dict["overall_avg"], mae_f1_dict["overall_avg"] = mae_regression_overall_avg, mae_f1_overall_avg
    acc_dict["across_categories_avg"], f1_dict["across_categories_avg"] = [acc_avg_across_categories,
                                                                           f1_avg_across_categories]
    mae_regression_dict["across_categories_avg"], mae_f1_dict["across_categories_avg"] = \
        [mae_regression_avg_across_categories, mae_f1_avg_across_categories]
    top11_regression_dict["across_categories_avg"], top11_f1_dict["across_categories_avg"] = \
        [top11_regression_avg_across_categories, top11_f1_avg_across_categories]

    compare_metrics_per_category_dict = combineMetricsPerCategory(
        wanted_categories_keys=wanted_categories, f1=f1_dict, mae=mae_regression_dict, mae_f1=mae_f1_dict, top11=top11_regression_dict, top11_f1=top11_f1_dict)
    # compare_metrics_per_category_dict = combineMetricsPerCategory(wanted_regression_keys=regression_keys_list, f1=f1_dict, top11=top11_regression_dict, top11_f1=top11_f1_dict)
    table_test = createTableList(f1=f1_dict, mae=mae_regression_dict,
                                 mae_f1=mae_f1_dict, top_11=top11_regression_dict, top11_f1=top11_f1_dict)

    acc_dict = append_suffix(acc_dict, "_acc")
    f1_dict = append_suffix(f1_dict, "_f1")
    mae_regression_dict = append_suffix(mae_regression_dict, "_mae_regression")
    mae_f1_dict = append_suffix(mae_f1_dict, "_mae_f1")
    top11_regression_dict = append_suffix(
        top11_regression_dict, "_top11_regression")
    top11_f1_dict = append_suffix(top11_f1_dict, "_top11_f1")

    return acc_dict, f1_dict, mae_regression_dict, mae_f1_dict, top11_regression_dict, top11_f1_dict, compare_metrics_per_category_dict, table_test


def combineRegressionAndF1Dicts(regression_dict, f1_dict):
    # ORDER important: keys which exist in both are taken from the second dict (regression one) and replace the classification ones!
    return {**f1_dict, **regression_dict}


def compute_category_avgs(metric_dict, use_extended_wrapper):
    category_dict = {}
    if use_extended_wrapper:
        summary_key_dictionary = summary_key_dict_extended
    else:
        summary_key_dictionary = summary_key_dict

    print("\nCategories are:")
    for category_name, category_keys in summary_key_dictionary.items():
        category_values = [
            v for k, v in metric_dict.items() if k in category_keys]
        category_values_names = [
            k for k, v in metric_dict.items() if k in category_keys]
        if len(category_values) < 1:
            continue
        category_mean = np.mean(category_values)
        category_dict[category_name + "_avg"] = category_mean
        print(category_name)
        for v in category_values_names:
            print("\t" + v)

    return category_dict
