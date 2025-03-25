import tqdm
import os
import random
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

"""
To view TensorBoard:
Open a terminal and run: tensorboard --logdir=logs
Open your browser and go to http://localhost:6006/.
"""

class TrainingArgs:
    def __init__(self,
                do_train:bool=False,
                do_eval:bool=False,
                do_infer:bool=False,
                train_dataset=None,
                eval_dataset=None,
                learning_rate=1e-3,
                batch_size=4,
                eval_batch_size=2,
                weight_decay:float=0.01,
                adam_epsilon:float=1e-6,
                max_step:int=1000,
                warmup_steps:int=5,
                save_steps:int=200,
                eval_steps:int=50,
                infer_steps:int=50,
                save_total_limit:int=5,
                output_dir:str='./base_model',
                logging_dir:str='./log',
                **kwargs
                ):
        self.do_train=do_train
        self.do_eval=do_eval
        self.do_infer=do_infer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.learning_rate = learning_rate
        self.batch_size=batch_size
        self.eval_batch_size=eval_batch_size
        self.weight_decay =weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_step=max_step
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.infer_steps=infer_steps
        self.save_total_limit=save_total_limit
        self.output_dir = output_dir
        self.logging_dir=logging_dir
        self.do_train = True if self.train_dataset!=None else self.do_train
        self.do_eval = True if self.eval_dataset!=None else self.do_eval

        assert self.do_train==True, "This is trining pipeline."
        if self.do_eval:
            assert self.eval_dataset!=None, 'Please provide proper Evaluation Dataset.'
        if self.do_eval:
            assert self.train_dataset!=None, "Please provide proper Training Dataset."

class Trainer:
    def __init__(self,model,args,device,tokenizer,name='base_model'):
        self.model=model
        self.device=device
        self.args=args
        self.name=name
        self.tokenizer = tokenizer
        
        self.writer = SummaryWriter(self.args.logging_dir)
    
    # def __del__(self):
    #     self.writer.close()

    def save_model(self,model,optimizer,step,loss):
        os.makedirs(self.args.output_dir,exist_ok=True)
        state_dict = dict(
            model = model.state_dict(),
            optimizer = optimizer.state_dict(),
            step=step,
            loss=loss
        )
        torch.save(state_dict,os.path.join(self.args.output_dir,f"{self.name}_e{step}.pt"))
        print(f'\nStep {step} | Model Saved to : {os.path.join(self.args.output_dir,f"{self.name}_e{step}.pt")}')
        return
    
    def eval(self,model,loss,data_loader):
        val_loss = 0 
        val_loader = tqdm.tqdm(data_loader)
        val_loader.set_description("Evaluating : ")
        with torch.no_grad():
            for masked_tokens,attention_mask,labels in val_loader:
                masked_tokens = masked_tokens.to(self.device)
                labels = labels.to(self.device)
                outputs = model(masked_tokens, labels=labels)
                val_loss += outputs.loss.item()

                del masked_tokens
                del labels
                del outputs

        print(f"\nVal Loss: {val_loss / len(data_loader)}")
        return val_loss/len(data_loader)

    def inference(self,step):
        self.model.eval()
        with torch.no_grad():
            len_dataset = len(self.args.eval_dataset)
            dataset_index = random.randint(0,len_dataset)
            input_ids,attention_mask,labels = self.args.eval_dataset[dataset_index]
            input_ids = input_ids.unsqueeze(0).to(self.device)

            pred = self.model(input_ids)[0].cpu()
            final_input_ids = input_ids.cpu().clone()

            mask_token_index = torch.where(final_input_ids==self.tokenizer.mask_token_id)[1]
            predicted_token_id = pred[0, mask_token_index, :].argmax(dim=-1)
            self.writer.add_text("Inference", f"Org  : {str(labels[mask_token_index].tolist())}\nPred : {str(predicted_token_id.tolist())}", step)

            del input_ids

    def train(self):
        train_loader = torch.utils.data.DataLoader(self.args.train_dataset,batch_size=self.args.batch_size,shuffle=True)
        
        if self.args.do_eval:
            eval_loader = torch.utils.data.DataLoader(self.args.eval_dataset,batch_size=self.args.eval_steps,shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.args.learning_rate,eps=self.args.adam_epsilon,weight_decay=self.args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.max_step
        )
        self.model=self.model.to(self.device)
        self.model.train()

        step = 0
        with tqdm.tqdm(total=self.args.max_step) as pbar:
            while step <self.args.max_step:
                for idx,(masked_token,attention_mask,labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    masked_token=masked_token.to(self.device)
                    labels=labels.to(self.device)
                    outputs = self.model(masked_token,labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()


                    del masked_token
                    del labels
                    del outputs

                    if step>0: 
                        # Saving
                        if step%self.args.save_steps==0:
                            self.save_model(self.model,optimizer,step,loss)
                            # torch.cuda.empty_cache()

                        # Evaluation

                        if self.args.do_eval and step%self.args.eval_steps==0:
                            self.model.eval()
                            eval_loss = self.eval(self.model,loss,eval_loader)
                            self.model.train()
                            
                            #Storing Validation Loss
                            self.writer.add_scalar("Loss/validation", eval_loss, step)

                        # if self.args.do_infer and step%self.args.infer_steps==0:
                        #     self.inference(step)


                    #Storing Training Loss
                    self.writer.add_scalar("Loss/train", loss.item(), step)
                    
                    step+=1

                    #Decoration
                    if step==self.args.max_step:
                        break
                    pbar.update(1)
                    pbar.set_postfix({'Loss':loss.item()})
        self.writer.close()
                    
