from Utils.tools import evaluate_bleu

references = ["我喜欢你的声音<\s>","这并不好<\s>"]
hypotheses= ["我喜a欢你<\s>","你很差<\s>"]
evaluate_bleu.compute_bleu(hypotheses,references )