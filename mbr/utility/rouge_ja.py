from evaluate import load

from utility.utility_class import UtilityFunction, MeCabTokenizer


class ROUGELJA(UtilityFunction):
    def __init__(self):
        self.similarity = load("rouge")
        self.mecab_tokenizer = MeCabTokenizer()

    def compute_similarity(self, hyp, ref, src):
        scores = self.similarity.compute(
            predictions=hyp,
            references=ref,
            use_stemmer=False,
            tokenizer=self.mecab_tokenizer.tokenize,
            use_aggregator=False,
        )["rougeL"]
        return scores
