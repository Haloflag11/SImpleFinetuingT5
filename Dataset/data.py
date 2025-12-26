from torch.utils.data import Dataset,DataLoader

import json

class RuQG(Dataset):
    def __init__(self,data_path,tokenizer,max_source_length=512, max_target_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data = []

        seen_questions = set()
        with open(self.data_path,'rt') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    question = item["question"]
                    if question not in seen_questions:
                        seen_questions.add(question)
                        self.data.append(item)
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    continue
        print(f'Data Length: {len(self.data)}')

    def __len__(self):
        return len(self.data)


    def __getitem__(self,idx):
        item=self.data[idx]
        input_text=f"question:{item['question']} context:{item['context']}"
        target_text=f"answer:{item['answer']}"

        #Tokenize
        model_inputs=self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            truncation=True,
            padding=False, #在collate_fn中处理
            return_tensors="pt",
        )
        labels=self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {
            'input_ids': model_inputs['input_ids'].squeeze(0),
            'attention_mask': model_inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0)
        }

