import os


class Emote_Config:
    def __init__(self, finetune, model_name, portion, seed, hfpath):
        
        self.finetune = finetune
        self.model_name = model_name
        self.portion = portion
        self.seed = seed
        self.hfpath = hfpath

        self.train_csv_fp = 'benchmark_data/exp-entailment/train.csv'
        self.val_csv_fp = 'benchmark_data/exp-entailment/val.csv'
        self.test_csv_fp = 'benchmark_data/exp-entailment/test.csv'

        # We choose popular models (high downloads) that are trained on MNLI
        if self.model_name == 'bert-base':
            self.model_path = "WillHeld/bert-base-cased-mnli"
        elif self.model_name == 'roberta-base':
            self.model_path = "WillHeld/roberta-base-mnli"
        elif self.model_name == 'roberta-large':
            self.model_path = 'FacebookAI/roberta-large-mnli'
        elif self.model_name == 'bart-large':
            self.model_path = 'facebook/bart-large-mnli'
        
        self.strategy_map = {
            'Direct': 0,
            'Metaphorical': 1,
            'Semantic list': 2,
            'Reduplication': 3,
            'Single': 4,
            'Others': 5,
            'Negatives': 6
        }

        self.bs = 16
        self.max_epoch = 10
        
        # Different models have different indices for entailment, neutral and contradiction
        # References:
        # https://huggingface.co/WillHeld/bert-base-cased-mnli/blob/main/config.json
        # https://huggingface.co/WillHeld/roberta-base-mnli/blob/main/config.json
        # https://huggingface.co/FacebookAI/roberta-large-mnli/blob/main/config.json
        # https://huggingface.co/facebook/bart-large-mnli/blob/main/config.json 
        self.entailment_idx, self.neutral_idx, self.contradiction_idx = None, None, None
        if self.model_name in ['bert-base', 'roberta-base']:
            self.entailment_idx = 0
            self.neutral_idx = 1
            self.contradiction_idx = 2
        elif self.model_name in ['roberta-large', 'bart-large']:
            self.entailment_idx = 2
            self.neutral_idx = 1
            self.contradiction_idx = 0

        self.unsup_dir = 'benchmark_data/results/TE-unsup'
        # check if the directory exists
        if not os.path.exists(self.unsup_dir):
            os.makedirs(self.unsup_dir)
        
        self.sft_dir = 'benchmark_data/results/TE-finetune/{}'.format(self.model_name)
        # check if the directory exists
        if not os.path.exists(self.sft_dir):
            os.makedirs(self.sft_dir)
        
        self.model_save_dir = '{}/emote-{}'.format(self.hfpath, self.model_name)
