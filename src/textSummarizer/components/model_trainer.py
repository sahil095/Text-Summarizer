from transformers import (
                    AutoTokenizer, 
                    AutoModelForSeq2SeqLM, 
                    Trainer, 
                    TrainingArguments, 
                    DataCollatorForSeq2Seq)

from textSummarizer.entity import ModelTrainerConfig
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model = model_pegasus)
        # loading data
        model = nn.DataParallel(model_pegasus)
        model.to(device)
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # dataset_samsum_pt = dataset_samsum_pt.to(device)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=12, 
            warmup_steps=500,
            per_device_train_batch_size=12, 
            per_device_eval_batch_size=8,
            learning_rate=0.0001,
            weight_decay=0.01, 
            eval_strategy='epoch', 
            eval_steps=2, 
            fp16=True
        ) 

        
        trainer = Trainer(
            model = model_pegasus,
            args = trainer_args,
            tokenizer = tokenizer,
            data_collator = seq2seq_data_collator,
            train_dataset = dataset_samsum_pt["train"],
            eval_dataset = dataset_samsum_pt["validation"]  
        )

        trainer.train()

        ### save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))

        ### save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))