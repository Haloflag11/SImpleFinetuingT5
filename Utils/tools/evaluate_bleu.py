import json
import os.path
from collections import defaultdict

import tokenizers
from sacrebleu import corpus_bleu
import pandas as pd


def group_references_by_question(file_path='/tmp/T5/datasets/DuReaderQG/dev.json'):
    """
    根据问题对答案进行分组，生成BLEU可用的references格式
    确保references与hypotheses逐句对齐

    Args:
        file_path: 数据文件路径

    Returns:
        references: BLEU可用的references格式，包含三个参考集
    """
    # 读取并解析数据
    with open(file_path, "r", encoding="utf-8") as f:
        data_lines = f.readlines()

    # 收集所有数据点
    data_points = []
    for line in data_lines:
        if line.strip():
            data = json.loads(line.strip())
            data_points.append(data)

    # 按问题分组答案
    question_groups = defaultdict(list)
    question_order = []  # 保持问题出现的顺序

    for data in data_points:
        question = data["question"]
        answer = data["answer"]
        if question not in question_groups:
            question_order.append(question)  # 记录第一次出现的问题顺序
        question_groups[question].append(answer)

    # 构建三个参考集
    ref1 = []  # 第一个回答
    ref2 = []  # 第二个回答
    ref3 = []  # 第三个回答

    for question in question_order:
        answers = question_groups[question]

        # 第一个参考：取第一个回答，如果存在
        ref1.append(answers[0] if len(answers) > 0 else "")

        # 第二个参考：取第二个回答，如果存在
        ref2.append(answers[1] if len(answers) > 1 else "")

        # 第三个参考：取第三个回答，如果存在
        ref3.append(answers[2] if len(answers) > 2 else "")

    # 确保三个参考集长度相同（等于问题数量）
    references = [ref1, ref2, ref3]

    return references

def compute_bleu(references,hypothesis,save_root='/tmp/T5/results'):
    print('Computing BLEU...')
    os.makedirs(save_root,exist_ok=True)

    def clean_text(text):
        # 去掉 "answer:" 前缀，如果存在的话
        if isinstance(text, str) and text.startswith('answer:'):
            return text.replace('answer:', '', 1).strip()
        return text

    # 清洗 hypotheses
    hypotheses = [clean_text(hyp) for hyp in hypothesis]
    # references=[[clean_text(ref) for ref in references]]
    scores=corpus_bleu(
        hypotheses=hypotheses, references=references,
        smooth_method='exp',
        force=False,  # 不强制长度匹配
        lowercase=False,  # 不转为小写
        tokenize='char'
   )

    precisions=scores.precisions

    res = {
        'bleu1': precisions[0],
        'bleu2': 0.5*precisions[1]+0.5*precisions[0],
        'bleu3': precisions[2]/3.0+precisions[1]/3.0+precisions[0]/3.0,
        'bleu4': scores.score
    }
    #Debug
    res_df=pd.DataFrame([res])
    file_name='bleu.csv'
    save_path = os.path.join(save_root,file_name)
    res_df.to_csv(save_path, index=False)
    print(f'BLEU Evalutation Done, saved to {save_path}')
    return res

#Test
# if __name__ == '__main__':
#     references = group_references_by_question()
#     print(references)
#     score,pre=compute_bleu(references=references,hypothesis=hyps,save_root='./')
#     print(score['bleu4'],score['bleu1'],score['bleu2'],score['bleu3'])
#     print(pre)