import os
import pandas as pd
import pyarrow.parquet as pq
import csv
import codecs
import torch
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import GPT2TokenizerFast,PreTrainedTokenizerFast,RobertaTokenizerFast

# from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import DataCollatorForLanguageModeling

from mlm_dataset import MLMDataset,MLMDatasetHF
from trainer import TrainingArgs,Trainer

# os.environ["CUDA_VISIBLE_DEVICES"]=","
# torch.cpu.set_device(0)

#GPU
torch.cuda.set_device(0)
torch.cuda.current_device()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device ',device)

#----------------------------Defination-------------------------
wikitext_dataset_pth="C:\\Users\\Pankaj Deb Roy\\Documents\\DeepLearning\\Dataset\\wikitext-103-raw-v1"
NX=12
MAX_LEN= 256

tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base",
                                                pad_tok = '<pad>',
                                                bos_tok = "<s>",
                                                eos_tok = "</s>",
                                                unk_token = "<unk>",
                                                mask_tok = "<mask>"
                                                )

model_config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=MAX_LEN,
    num_attention_heads=12,
    num_hidden_layers=NX,
    type_vocab_size=1
)
model = RobertaForMaskedLM(config=model_config)
print('Num parameters: ',model.num_parameters())



train_dataset = MLMDataset(os.path.join(wikitext_dataset_pth,'test_wiki-104.csv'), tokenizer,max_seq_len=MAX_LEN)
eval_dataset = MLMDataset(os.path.join(wikitext_dataset_pth,'val_wiki-104.csv'), tokenizer,max_seq_len=MAX_LEN)

training_args = TrainingArgs(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    do_infer=True,
    # load_model_dir="./mlm_roberta/base_model_e900.pt",
    output_dir="./mlm_roberta",
    learning_rate=1e-5,
    batch_size=16,
    eval_batch_size=8,
    weight_decay=0.01,
    adam_epsilon=1e-6,
    max_step=10000,
    warmup_steps=10,
    save_steps=100,
    eval_steps= 50,
    infer_steps=50,
    logging_dir='./logs',
)

trainer = Trainer(model=model,tokenizer=tokenizer,args=training_args,device=device)

trainer.train()



# #Training With Huggingface
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=True,mlm_probability=0.15)
# print("Training Starting . . . . . ")
# training_args = TrainingArguments(
#                                 output_dir="./mlm_roberta",
#                                 # eval_strategy="steps",
#                                 prediction_loss_only=True,
#                                 per_device_train_batch_size= 32,
#                                 # per_device_eval_batch_size=1,
#                                 # eval_accumulation_steps = 200,
#                                 weight_decay=0.01,
#                                 adam_epsilon=1e-6,
#                                 max_steps=10_000,
#                                 warmup_steps=1,
#                                 save_steps=200,
#                                 save_total_limit=5,
#                                 # eval_steps= 100,
#                                 logging_dir='./logs',
#                                 )
# print(f'Device : ',training_args.device)
# trainer = Trainer(
#                 model=model,
#                 args=training_args,
#                 data_collator=data_collator,
#                 train_dataset=train_dataset,
#                 # eval_dataset=eval_dataset,
#                 )

# trainer.train()