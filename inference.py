import os

from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq
from peft import PeftModel
import torch

from Dataset.data import RuQG

from Utils.sft import Trainer as SFTTrainer
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

def load_best_lora_model(original_model_path, lora_checkpoint_path, device="cuda"):
    """
    加载最佳LoRA检查点进行推理

    Args:
        original_model_path: 原始T5模型路径
        lora_checkpoint_path: LoRA检查点路径（包含adapter文件）
        device: 推理设备

    Returns:
        model: 加载好的模型
        tokenizer: 分词器
    """

    print("Loading original T5 model...")
    # 1. 加载原始模型
    model = T5ForConditionalGeneration.from_pretrained(
        original_model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        low_cpu_mem_usage=True

    )

    print("Loading LoRA adapters...")
    # 2. 加载LoRA权重到原始模型
    model = PeftModel.from_pretrained(
        model,
        lora_checkpoint_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )


    model = model.merge_and_unload()



    tokenizer = T5Tokenizer.from_pretrained(original_model_path)

    print("Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")

    return model, tokenizer





# 运行推理
if __name__ == "__main__":
    model,tokenizer=load_best_lora_model(original_model_path='/root/autodl-tmp/models/mengzi-t5-base',lora_checkpoint_path='/tmp/T5/Result/best_checkpoint')
    train_set_loader, dev_set_loader = load_data(tokenizer)

    #Trainer as Infernecer
    trainer=SFTTrainer(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        train_loader=train_set_loader,
        dev_loader=dev_set_loader,
        test_loader=None,
        device="cuda:0",
        epochs=None,
        tokenizer=tokenizer,
    )
    #
    # trainer.evaluate(trainer.model,epoch=None,data_loader_type='Train',cal_bleu=True)
    score,_=trainer.evaluate(trainer.model,epoch=None,data_loader_type='Dev',cal_bleu=True)
    print(
        f" BLEU-1: {score['bleu1']:.4f}, BLEU-2: {score['bleu2']:.4f}, BLEU-3: {score['bleu3']:.4f} , BLEU-4: {score['bleu4']:.4f}")

