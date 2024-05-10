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


class EmoteTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, new_emote_config, tokenizer):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.new_emote_config = new_emote_config
        self.tokenizer = tokenizer

        self.batch_size = self.new_emote_config.bs
        self.model_name = self.new_emote_config.model_name
        self.portion = self.new_emote_config.portion
        self.seed = self.new_emote_config.seed
        self.strategy_map = self.new_emote_config.strategy_map
        self.strategy_map_inverted = {v: k for k, v in self.strategy_map.items()}
        self.max_epochs = 10

        self.loss_fn = BCEWithLogitsLoss()

        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.best_val_loss = np.inf
        self.best_val_acc = 0
        self.best_epoch = -1

        self.save_dir = self.new_emote_config.sft_dir
        self.fp_txt = '{}/portion_{}_seed_{}.txt'.format(self.save_dir, self.portion, self.seed)
        self.fp_csv = '{}/portion_{}_seed_{}.csv'.format(self.save_dir, self.portion, self.seed)
        self.fp_csv_best = '{}/portion_{}_seed_{}_best.csv'.format(self.save_dir, self.portion, self.seed)

    def train(self, epochs):
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            
            self._run_epoch(epoch, train=True)
            val_acc, val_acc_strategy = self._run_epoch(epoch=epoch, train=False, loader=self.val_loader)
            test_acc, test_acc_strategy = self._run_epoch(epoch=epoch, train=False, loader=self.test_loader)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                small_dict = {}
                small_dict['Overall'] = test_acc
                small_dict['Direct'] = test_acc_strategy[0]
                small_dict['Metaphorical'] = test_acc_strategy[1]
                small_dict['Semantic list'] = test_acc_strategy[2]
                small_dict['Reduplication'] = test_acc_strategy[3]
                small_dict['Single'] = test_acc_strategy[4]
                small_dict['Negative'] = test_acc_strategy[5]
                # save it into a csv
                with open(self.fp_csv_best, 'w') as f:
                    f.write('strategy,accuracy\n')
                    for key in small_dict.keys():
                        f.write('{},{}\n'.format(key, small_dict[key]))

            # if epoch is 0, then save the model
            if epoch == 0 and self.portion == 1: ## we now only save fully trained models
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_save.save_pretrained("{}/seed_{}_epoch_{}.pt".format(self.new_emote_config.model_save_dir, self.seed, epoch))
                self.tokenizer.save_pretrained("{}/seed_{}_epoch_{}.pt".format(self.new_emote_config.model_save_dir, self.seed, epoch))
                print("Saving model checkpoint to %s" % "{}/seed_{}_epoch_{}.pt".format(self.new_emote_config.model_save_dir, self.seed, epoch))
            if epoch > 0 and val_acc > self.best_val_acc and self.portion == 1:
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_save.save_pretrained("{}/seed_{}_epoch_{}.pt".format(self.new_emote_config.model_save_dir, self.seed, epoch))
                self.tokenizer.save_pretrained("{}/seed_{}_epoch_{}.pt".format(self.new_emote_config.model_save_dir, self.seed, epoch))
                print("Saving model checkpoint to %s" % "{}/seed_{}_epoch_{}.pt".format(self.new_emote_config.model_save_dir, self.seed, epoch))

        # Print the epoch with the best validation loss
        print(f'Best Epoch: {self.best_epoch}')

        # Print the best val accuracy and its epoch number
        print(f'Best Val Accuracy: {self.best_val_acc} at Epoch {self.best_epoch}')
        # Write it into fp_txt
        with open(self.fp_txt, 'a') as f:
            f.write(f'Best Val Accuracy: {self.best_val_acc} at Epoch {self.best_epoch}\n')


    def _run_epoch(self, epoch, train, loader=None):
        if train:
            self.model.train()
            loader = self.train_loader
            str_ = 'Train'
        else:
            self.model.eval()
            str_ = 'Valid' if loader == self.val_loader else 'Test'

        total_loss = 0
        predictions, true_labels, strategy_types = [], [], []
        strategies_correct = [0]*7
        strategies_total = [0]*7

        for batch in loader:
            # Separate strategies from the rest of the batch data
            strategies = batch.pop('strategies').to(self.device)

            # Move batch (without strategies) to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.set_grad_enabled(train):
                outputs = self.model(**batch)
                loss = self.loss_fn(outputs.logits, torch.nn.functional.one_hot(batch['labels'].long(), num_classes=2).float())

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs.logits, 1)
            predictions.extend(preds.cpu().numpy().tolist())
            true_labels.extend(batch['labels'].cpu().numpy().tolist())
            strategy_types.extend(strategies.cpu().numpy().tolist())

            current_true_labels = batch['labels'].cpu().numpy().tolist()
            current_strategies = strategies.cpu().numpy().tolist()

            for i in range(len(preds)):
                strategies_total[current_strategies[i]] += 1
                if preds[i] == current_true_labels[i]:
                    strategies_correct[current_strategies[i]] += 1

        avg_loss = total_loss / len(loader)
        print(f'{str_} Loss: {avg_loss}')
        print(classification_report(true_labels, predictions, zero_division=1, digits=4))

        if str_ in ['Valid', 'Test']:
            with open(self.fp_txt, 'a') as f:
                # Write epoch number and str_ into fp_txt
                f.write('Epoch: {}, {}\n'.format(epoch+1, str_))
                f.write(classification_report(true_labels, predictions, zero_division=1, digits=4))
                f.write('\n')
        
        if str_ in ['Valid', 'Test']:
            with open(self.fp_csv, 'a') as f:
                f.write('\n*** start *** {}\n'.format(epoch+1))
                f.write('Epoch: {}\n'.format(epoch+1))
                f.write('{}\n'.format(str_))

                f.write('{},{},{},{}\n'.format('idx', 'pred', 'true', 'strategy'))
                for idx in range(len(predictions)):
                    f.write('{},{},{},{}\n'.format(idx, predictions[idx], true_labels[idx], strategy_types[idx]))
            print('write predictions into {}'.format(self.fp_csv))

        # Get accuracy score overall
        overall_accuracy = round(np.sum(np.array(predictions) == np.array(true_labels)) / len(true_labels)  * 100, 1)
        if str_ in ['Valid', 'Test']:
            with open(self.fp_txt, 'a') as f:
                f.write(f'{str_} accuracy: {overall_accuracy}\n\n')
                # Then write header 'strategy, accuracy, correct count / total number' into fp_txt
                f.write('{}, {}, {}\n'.format('strategy', 'accuracy', 'correct count / total number'))

        strategy_acc = []
        # Finally, after the loop, print the accuracies for each strategy:
        for i in range(len(self.strategy_map)):
            if strategies_total[i] != 0:
                # print accuracy for strategy i, with its correct count and totol number of the strategy
                accuracy = round(strategies_correct[i]/strategies_total[i] * 100, 1)
                strategy_acc.append(accuracy)
                print(f'Strategy {i} {self.strategy_map_inverted[i]} accuracy: {accuracy}, {strategies_correct[i]} / {strategies_total[i]}')
                # If str_ is in ['Valid', 'Test'], then write the accuracy for strategy i into fp_txt
                if str_ in ['Valid', 'Test']:
                    with open(self.fp_txt, 'a') as f:
                        f.write(f'{i}, {accuracy}, {strategies_correct[i]} / {strategies_total[i]}\n')
                        if i == len(self.new_emote_config.strategy_map) - 1:
                            f.write('\n')
            else:
                print(f'No instances of strategy {i} {self.strategy_map_inverted[i]} found')

        return overall_accuracy, strategy_acc
