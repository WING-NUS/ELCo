import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW
from emote_config import Emote_Config


class UnsupervisedEval:
    def __init__(self, model, test_loader, device, split_name, new_emote_config):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.split_name = split_name
        self.new_emote_config = new_emote_config
        self.model_name = new_emote_config.model_name
        self.strategy_map = new_emote_config.strategy_map
        self.strategy_map_inverted = {v: k for k, v in self.strategy_map.items()}
        self.fp_txt = '{}/{}_{}.txt'.format(new_emote_config.unsup_dir, self.model_name, split_name)
        self.fp_all_res = '{}/{}_{}.csv'.format(new_emote_config.unsup_dir, self.model_name, split_name)

    def unsupervised_evaluation(self):
        # freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        directory = self.new_emote_config.unsup_dir

        fp_txt = '{}/{}_{}.txt'.format(directory, self.model_name, self.split_name)
        fp_all_res = '{}/{}_{}.csv'.format(directory, self.model_name, self.split_name)
        # Initialize lists to store all softmax, labels, and strategies
        all_softmax = []
        all_labels = []
        all_strategies = []

        # Loop over each batch
        batch_idx = 0

        for batch in self.test_loader:
            # Extract the strategies from the batch and move them to CPU
            strategies = batch.pop('strategies').cpu().numpy()
            all_strategies.append(strategies)
            print('batch_idx: ', batch_idx)
            # Move the batch to the device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Extract the labels from the batch and move them to CPU
            labels = batch['labels'].cpu().numpy()
            all_labels.append(labels)
            # Get the model's output
            outputs = self.model(**batch)
            # Get the logits
            logits = outputs.logits
            # Calculate the softmax
            softmax = torch.nn.functional.softmax(logits, dim=-1)
            # Move the softmax to CPU and convert to numpy array
            softmax = softmax.cpu().detach().numpy()
            # Append the batch softmax to the list
            all_softmax.append(softmax)
            batch_idx += 1

        # Concatenate all softmax, labels, and strategies
        all_softmax = np.concatenate(all_softmax, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_strategies = np.concatenate(all_strategies, axis=0)

        # Initialize an empty list to store the results
        results = []

        # Iterate over all instances in all_softmax
        for instance in all_softmax:
            if instance[self.new_emote_config.entailment_idx] > instance[self.new_emote_config.neutral_idx] + instance[self.new_emote_config.contradiction_idx]:
                results.append(1)
            else:
                results.append(0)

        # Generate the classification report for all data
        report = classification_report(all_labels, results, digits=4)
        print(report)

        # save the report into fp_txt
        with open(fp_txt, 'w') as f:
            f.write(report)
            f.write('\n')

        all_res_list = []
        for idx, res in enumerate(results):
            all_res_list.append((idx, res))

        with open(fp_all_res, 'w') as f:
            # First write header 'idx, pred, true' into fp_all_res
            f.write('{},{},{},{}\n'.format('idx', 'pred', 'true', 'strategy'))
            for idx, res in all_res_list:
                f.write('{},{},{},{}\n'.format(idx, res, all_labels[idx], all_strategies[idx]))
        print('write all_res_list into {}'.format(fp_all_res))
        
        with open(fp_txt, 'a') as f:
            f.write('{}, {}, {}\n'.format('strategy', 'accuracy', 'correct count / total number'))
        # Now, generate classification reports separately for each strategy
        # Put accuracy score into a json and print it as csv
        accuracy_dict = {}
        for i in range(len(self.strategy_map)):
            mask = (all_strategies == i)
            # check if there is no instance of strategy i
            if np.sum(mask) == 0:
                print(f'No instances of strategy {i} found')
                continue
            # calculate the accuracy for strategy i
            accuracy = round(np.sum(np.array(results)[mask] == all_labels[mask]) / len(all_labels[mask])  * 100, 1)
            # print strategy, accuracy (in 67.8 style)  and correct count and total number of the strategy
            print(f'Strategy {i} accuracy: {accuracy}, {np.sum(np.array(results)[mask] == all_labels[mask])} / {len(all_labels[mask])}')
            # save i, accuracy, correct count and total number of the strategy into fp_txt
            with open(fp_txt, 'a') as f:
                f.write(f'{i}, {accuracy}, {np.sum(np.array(results)[mask] == all_labels[mask])} / {len(all_labels[mask])}\n')
            accuracy_dict[i] = accuracy
        # Get overall accuracy score
        overall_accuracy = round(np.sum(np.array(results) == all_labels) / len(all_labels)  * 100, 1)
        # print overall accuracy score
        print(f'Overall accuracy: {overall_accuracy}')

        # save overall accuracy score into fp_txt
        with open(fp_txt, 'a') as f:
            f.write(f'Overall accuracy: {overall_accuracy}\n')
        
        # print the saving fp
        print('write accuracy_dict into {}'.format(fp_txt))
        print('predicted labels saved into {}'.format(fp_all_res))
