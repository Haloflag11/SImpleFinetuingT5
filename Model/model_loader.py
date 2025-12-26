from torch import nn
from transformers import T5ForConditionalGeneration,T5ForQuestionAnswering

class T5forQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_encoder=T5ForConditionalGeneration.from_pretrained('t5-small')
