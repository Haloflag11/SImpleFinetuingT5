import os
import time
from typing import Tuple

import numpy as np

from Utils.tools.leakage import check_data_leakage
import torch
from datasets import tqdm
from matplotlib import pyplot as plt

from accelerate import Accelerator
from transformers.utils import logging as logging_utils
from transformers import T5ForConditionalGeneration
from torch.utils.data import DataLoader
from Dataset.data import RuQG
from transformers.models import AutoModel,T5Tokenizer
from transformers import DataCollatorForSeq2Seq
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR ,LinearLR, SequentialLR

from peft import LoraConfig,get_peft_model,TaskType

from Utils.sft import Trainer as SFTTrainer



#加载数据
def load_data(tokenizer):
    dataset_root='/tmp/T5/datasets/DuReaderQG'
    train_set=RuQG(os.path.join(dataset_root,'train.json'),tokenizer=tokenizer)
    dev_set=RuQG(os.path.join(dataset_root,'dev.json'),tokenizer=tokenizer)
    print(f'Trainset:found {len(train_set)}, Devset:found {len(dev_set)}')

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True

    )
    train_set_loader=DataLoader(train_set,batch_size=16,shuffle=True,collate_fn=data_collator)
    dev_set_loader=DataLoader(dev_set,batch_size=16,collate_fn=data_collator)
    print("Load Complete")
    return train_set_loader, dev_set_loader

#加载模型
def load_model(model_path,device=None,tokenizer_only=False,use_lora=False):
    logging_utils.set_verbosity_debug()
    print(f"Loading model from {model_path}")
    try:
        print("Loading Tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        print("Tokenizer Loaded")

        print("Loading Model...")
        start = time.time()


        if not tokenizer_only:
            #没有自己写，T5ForConditionalGeneration已经有输出头了
            model=T5ForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto"
            )
            if use_lora:
                #  LoRA配置
                lora_config = LoraConfig(
                    r=4,
                    lora_alpha=16,
                    target_modules=[
                        "q", "v",
                        "k", "o"
                    ],
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM  # QUESTION_ANS

                )
                model=get_peft_model(model,lora_config)
                model.print_trainable_parameters()


            period=time.time() - start
            print(f"Model Loaded in {period:.2f} seconds")
            total_params = sum(p.numel() for p in model.parameters())#参数量
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)#可训练参数量

            print(f"📊 模型参数统计:")
            print(f"  总参数: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  冻结参数: {total_params - trainable_params:,}")

            print("Load Complete")
            return model, tokenizer
        else:
            return None,tokenizer
    except Exception as e:
        print(f"Error occurred while loading model: {e}")
        raise e




def plot_loss(loss_array,eval_loss_array):
    """
    绘制训练损失曲线

    Args:
        loss_array: 损失值列表或数组
        eval_loss_array:验证损失
    """
    # 转换为numpy数组
    loss_array = np.array(loss_array)
    eval_loss_array=np.array(eval_loss_array)

    # # 调试信息
    # print(f"Plotting loss curve with {len(loss_array)} points")
    # print(f"Loss array shape: {loss_array.shape}")
    # print(f"Eval loss array shape: {eval_array.shape}")
    # print(f"Loss range: {loss_array.min():.4f} ~ {loss_array.max():.4f}")

    # 确保是一维数组
    if loss_array.ndim > 1:
        loss_array = loss_array.squeeze()

    if eval_loss_array.ndim > 1:
        eval_loss_array = eval_loss_array.squeeze()

    if len(loss_array) == 0:
        print("Warning: loss_array is empty!")
        return

    # 创建保存目录
    save_path = '/tmp/T5/Result/plot_loss.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 绘制图形
    plt.figure(figsize=(12, 6))

    # 绘制原始损失曲线
    x_axis = range(len(loss_array))
    plt.plot(x_axis, loss_array, '-', color='c', linewidth=1, alpha=0.7, label='Step Loss')
    if len(eval_loss_array) > 0:
        eval_interval = max(1, len(loss_array) // len(eval_loss_array))
        x_eval = [eval_interval * (i + 1) for i in range(len(eval_loss_array))]
        plt.plot(x_eval, eval_loss_array, 'o-', color='magenta', linewidth=2,alpha=0.7,
                 markersize=6, label='Validation Loss')

    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.title('Training and Validayion Loss', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免内存泄漏

    print(f"Loss curve saved to: {save_path}")



if __name__=='__main__':
    model_path='/root/autodl-tmp/models/mengzi-t5-base'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model,tokenizer=load_model(model_path,device,tokenizer_only=False,use_lora=True)#使用LoRA

    train_set_loader, dev_set_loader=load_data(tokenizer)

    #检查数据

    train_data=train_set_loader.dataset[0]
    dev_data=dev_set_loader.dataset[0]
    print(f"shape of train_set{train_data['input_ids'].shape}, shape of dev_set{dev_data['input_ids'].shape}")
    # #检查泄露
    # check_data_leakage(train_loader=train_set_loader,val_loader=dev_set_loader)


    #optimizer and schedule
    optimizer=AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    cos_scheduler=CosineAnnealingLR(
        optimizer,
        T_max=908*10-800,
        eta_min=2e-6
    )
    warmup_scheduler = LinearLR(optimizer=optimizer,start_factor=0.1,end_factor=1.0,total_iters=800)
    lr_scheduler=SequentialLR(optimizer,schedulers=[warmup_scheduler,cos_scheduler],milestones=[800])

    #Train and eval
    save_root='/tmp/T5/Result'
    os.makedirs(save_root,exist_ok=True)
    save_path='/tmp/T5/Result/best_checkpoint'

    epochs =10
    #Early stop
    patience=3
    patience_counter=0

    #Trainer
    trainer=SFTTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_set_loader,
        dev_loader=dev_set_loader,
        test_loader=None,
        device=device,
        epochs=epochs,
        tokenizer=tokenizer,
    )


    best_loss=float('inf')
    best_bleu_score=0
    eval_loss=[]

    for epoch in range(epochs):
        if epoch >=0:#可以控制隔多少epoch评估一次，这里每个都评估
            cal_bleu=True
        else:
            cal_bleu=False
        trainer.train(epoch=epoch,evaluate_when_training=False)
        score,loss=trainer.evaluate(model=trainer.model,epoch=epoch,cal_bleu=cal_bleu)#不计算BLEU
        eval_loss.append(loss)
        if score['bleu4']>best_bleu_score:
            patience_counter = 0
            best_loss = loss
            best_bleu_score=score['bleu4']
            trainer.model.save_pretrained(save_path)
            print(f"best_checkpoint_saved:{save_path}")
        else:
            patience_counter+=1
            if patience_counter>=patience:
                print(f"Early Stopping...")
                break
    plot_loss(loss_array=trainer.loss_array,eval_loss_array=eval_loss)









