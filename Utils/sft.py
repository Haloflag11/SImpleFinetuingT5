import json
import os
from typing import Optional

import numpy as np
import peft
import torch


from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from Utils.tools.evaluate_bleu import compute_bleu, group_references_by_question


class Trainer:
    def __init__(
    self,
    model:T5ForConditionalGeneration | peft.PeftModel,
    train_loader:DataLoader|None,
    dev_loader:DataLoader|None,
    test_loader:DataLoader |None,
    optimizer:Optimizer,
    lr_scheduler:LRScheduler,
    device:torch.device,
    epochs:int,
    tokenizer:T5Tokenizer,

    ):
        self.tokenizer = tokenizer
        self.model=model
        self.train_loader=train_loader
        self.dev_loader=dev_loader
        self.test_loader=test_loader
        self.optimizer=optimizer
        self.device=device
        self.epochs=epochs
        self.lr_scheduler=lr_scheduler

        self.loss_array=[]

    def train(self,epoch,evaluate_when_training=False):
        """

        :param epoch: 迭代周期
        :param evaluate_when_training:是否在训练中途评估
        :return: avg_loss,self.model,loss_array
        """
        save_path='/tmp/T5/results'
        os.makedirs(save_path, exist_ok=True)
        total_steps = len(self.train_loader)
        progress_bar=tqdm(total=total_steps)
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            if (batch_idx+1)%100 and evaluate_when_training:
                cal_bleu = True
            else:
                cal_bleu = False
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # 训练
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            loss=outputs.loss
            total_loss+=loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            self.loss_array.append(loss.item())
            avg_loss=total_loss/(batch_idx+1)
            progress_bar.set_description(
                f'Epoch {epoch+1}/{self.epochs} | loss: {avg_loss:>7f}'
            )
            progress_bar.update(1)

            if cal_bleu:
                self.model.eval()
                with torch.no_grad():
                    score, _ = self.evaluate(self.model, epoch, cal_bleu=cal_bleu)
                self.model.train()
                tqdm.write(
                    f"BLEU: {score['bleu']:.4f}, BLEU-1: {score['bleu1']:.4f}, BLEU-2: {score['bleu2']:.4f}, BLEU-3: {score['bleu3']:.4f} , BLEU-4: {score['bleu4']:.4f}")

        return avg_loss,self.model

    def evaluate(self,model:T5ForConditionalGeneration,epoch,data_loader_type='Dev',cal_bleu=False):
        """

        :param model:
        :param epoch:
        :param data_loader_type:训练时验证/评估验证/推理
        :param cal_bleu:
        :return:
        """
        assert data_loader_type in ('Train','Dev','Test'),"data_loader must be 'Train', 'Dev' or 'Test'"

        if data_loader_type=='Train':
            data_loader=self.train_loader
        elif data_loader_type=='Dev':
            data_loader=self.dev_loader
        elif data_loader_type=='Test':
            data_loader=self.test_loader
        if epoch is None:
            epoch=0
        save_root=f'/tmp/T5/results/eval_epoch_{epoch}/{data_loader_type}'
        os.makedirs(save_root, exist_ok=True)
        total_steps = len(data_loader)
        progress_bar = tqdm(total=total_steps)

        model.eval()
        total_loss=0
      #去重输出
        all_hypotheses =[]
        avg_loss=0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                if data_loader_type !='Test':
                    logits=model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask
                        )
                    loss = logits.loss
                    total_loss += loss.item()

                    avg_loss = total_loss/(batch_idx+1)


                if cal_bleu or data_loader_type=='Test' or data_loader_type=='Train':
                    outputs=model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        max_length=512,
                        num_beams=4,  # 束搜索，提高质量
                        early_stopping=True,
                        do_sample=False,
                        temperature=1.0,
                        decoder_start_token_id=self.tokenizer.pad_token_id,#T5的starttoken和padtoken是一个
                        pad_token_id=self.tokenizer.pad_token_id
                   )

                    decoded_predictions = self.tokenizer.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )


                    #存储推理结果0000 0
                    all_hypotheses.extend(decoded_predictions)
                progress_bar.set_description(f'Loss: {avg_loss:>7f}')  # batchsize:16
                progress_bar.update(1)
        all_references = group_references_by_question()  # 获取问题匹配的候选答案数据
        json_data = {
            'outputs': all_hypotheses,
            'labels': all_references
        }
        res_filename = os.path.join(save_root, 'results.json')
        with open(res_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        #计算BLEU
        if cal_bleu and len(all_hypotheses) > 0:

            tqdm.write(f"开始计算BLEU，样本数: {len(all_hypotheses)}")
            bleu_score = compute_bleu(references=all_references, hypothesis=all_hypotheses, save_root=save_root)
        else:
            bleu_score = 0.0

        tqdm.write(f"BLEU-1: {bleu_score['bleu1']:.4f}, BLEU-2: {bleu_score['bleu2']:.4f}, BLEU-3: {bleu_score['bleu3']:.4f} , BLEU-4: {bleu_score['bleu4']:.4f}")
        return bleu_score,avg_loss





