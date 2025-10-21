import numpy as np
from nltk.tokenize import ToktokTokenizer
from distinct_n.utils import ngrams
from jreadability import compute_readability

import torch
from evaluate import load
from torchmetrics.text.infolm import InfoLM
from transformers import (
    CLIPTextModel,
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoTokenizer,
)
from torch.nn.functional import cosine_similarity

import utility.utility_class as utility_class
from utility.utility_class import MeCabTokenizer
from utility.rouge_ja import ROUGELJA
from utility.metricx import METRICX
from utility.dsari import DSARI


def load_similarity(sim):
    if sim == "sentbert":
        return utility_class.SENTBERT()
    elif sim == "cliptext":
        return utility_class.CLIPTEXT()
    elif sim == "comet":
        return utility_class.COMET()
    elif sim == "comet20":
        return utility_class.COMET20()
    elif sim == "bertscore":
        return utility_class.BERTSCORE()
    elif sim == "bleurt":
        return utility_class.BLEURT()
    elif sim == "rouge":
        return utility_class.ROUGEL()
    elif sim == "rougeja":
        return ROUGELJA()
    elif sim == "sacrebleu":
        return utility_class.SACREBLEU()
    elif sim == "sacrebleuja":
        return utility_class.SACREBLEU(language="ja")
    elif "sfr2" in sim:
        return utility_class.SFR("Salesforce/SFR-Embedding-2_R")
    elif sim == "metricx_xl":
        return METRICX("google/metricx-23-xl-v2p0")
    elif sim == "metricx_xxl":
        return METRICX("google/metricx-23-xxl-v2p0")
    elif sim == "cer":
        return utility_class.CER()
    elif sim == "cer-neologdn":
        return utility_class.CER(normalizer="neologdn")
    elif sim == "cer-nonorm":
        return utility_class.CER(normalizer="none")
    elif sim == "wer":
        return utility_class.WER()
    elif sim == "wer-nonorm":
        return utility_class.WER(normalizer="none")
    else:
        raise ValueError(f"Invalid similarity function: {sim}")


def load_distance(sim, compute_similarity):
    if sim != "sacrebleu":

        def compute_distance(hyp, ref, src):
            return [1.0 - sim for sim in compute_similarity(hyp, ref, src)]
    else:
        # sacrebleu ranges (0, 100), so need to normalize it.
        def compute_distance(hyp, ref, src):
            return [1.0 - sim / 100.0 for sim in compute_similarity(hyp, ref, src)]

    return compute_distance


def load_evaluate(eval_func, sim, similarity):
    if eval_func == "bleurt":
        evaluator = load(eval_func, checkpoint="BLEURT-20")
    elif (eval_func == "comet") or (eval_func == "comet20"):
        from comet import download_model, load_from_checkpoint

        if eval_func == "comet":
            evaluator = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        else:
            evaluator = load_from_checkpoint(download_model("Unbabel/wmt20-comet-da"))
    elif eval_func == "clip":
        pass
    elif eval_func == "metricx":
        pass
    elif eval_func == "infolm":
        evaluator = InfoLM(
            "google/bert_uncased_L-2_H-128_A-2",
            information_measure="fisher_rao_distance",
            idf=False,
        )
    elif eval_func == "sentbert":
        pass
    elif eval_func == "jreadability":
        pass
    elif eval_func == "rougeja":
        evaluator = load("rouge")
    elif eval_func == "dsari":
        pass
    elif eval_func == "clair":
        pass
    elif "sacrebleu" in eval_func:
        evaluator = load("sacrebleu")
    elif eval_func in ["gender", "gender-gemma", "gender-gemma27", "jbbq-gemma"]:
        pass
    elif eval_func == "parse-answer":
        pass
    elif "ot-" in eval_func:
        pass
    elif (
        (eval_func == "cer")
        or (eval_func == "cer-neologdn")
        or (eval_func == "cer-nonorm")
    ):
        pass
    elif (eval_func == "wer") or (eval_func == "wer-nonorm") or (eval_func == "wer-th"):
        pass
    else:
        try:
            evaluator = load(eval_func)
        except:
            print("eval_func=", eval_func)
            pass

    if eval_func == "rouge":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[[ref]])["rougeL"]
    elif eval_func == "rougeja":
        mecab_tokenizer = MeCabTokenizer()

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(
                predictions=[hyp],
                references=[[ref]],
                use_stemmer=False,
                tokenizer=mecab_tokenizer.tokenize,
            )["rougeL"]
    elif eval_func == "sacrebleu":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref])["score"]
    elif eval_func == "sacrebleuja":
        # mecab_tokenizer = MeCabTokenizer()
        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(
                predictions=[hyp], references=[ref], tokenize="ja-mecab"
            )["score"]
    elif eval_func == "sacrebleuzh":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(
                predictions=[hyp], references=[ref], tokenize="zh"
            )["score"]
    elif eval_func == "bleurt":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref])["scores"][0]
    elif eval_func == "comet":

        def compute_evaluate(hyp, ref, src):
            d = {"src": src, "mt": hyp, "ref": ref}
            data = [d]
            model_output = evaluator.predict(data, progress_bar=False)
            return model_output.scores[0]
    elif eval_func == "comet20":

        def compute_evaluate(hyp, ref, src):
            d = {"src": src, "mt": hyp, "ref": ref}
            data = [d]
            model_output = evaluator.predict(data, progress_bar=False)
            return model_output.scores[0]
    elif eval_func == "infolm":

        def compute_evaluate(hyp, ref, src):
            return np.array(evaluator(hyp, ref)).item()
    elif eval_func == "meteor":

        def compute_evaluate(hyp, ref, src):
            scores = [
                evaluator.compute(predictions=[hyp], references=[r])["meteor"]
                for r in ref
            ]
            return max(scores)
    elif eval_func == "metricx":
        # metricx-xl
        evaluator = METRICX("google/metricx-23-xl-v2p0")

        def compute_evaluate(hyp, ref, src):
            scores = evaluator.compute_similarity([hyp], [ref], [src])
            return sum(scores) / len(scores)
    elif eval_func == "dsari":
        evaluator = DSARI(ngrams=4)

        def compute_evaluate(hyp, ref, src):
            scores = [evaluator.compute_similarity(hyp, r, src) for r in ref]
            return sum(scores) / len(scores)
    elif eval_func == "clip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/clip-vit-large-patch14"
        processor = CLIPProcessor.from_pretrained(model_id)

        model = CLIPModel.from_pretrained(model_id).to(device)
        evaluator = CLIPTextModel.from_pretrained(model_id).to(device)
        model.eval()
        evaluator.eval()

        def compute_evaluate(hyp, ref, src):
            with torch.no_grad():
                inputs = processor(
                    text=[hyp] + ref,
                    images=src,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to("cuda")

                text_embeddings = torch.flatten(
                    evaluator(inputs.input_ids.to(device))["last_hidden_state"], 1, -1
                )
                hyp_embeddings = text_embeddings[:1]
                ref_embeddings = text_embeddings[1:]
                text_scores = (
                    cosine_similarity(hyp_embeddings, ref_embeddings)
                    .cpu()
                    .detach()
                    .numpy()
                    .max()
                )
                # print('text_scores.shape=', text_scores.shape)

                img_inputs = processor(
                    text=hyp,
                    images=src,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to("cuda")
                img_outputs = model(**img_inputs)

                img_scores = np.squeeze(
                    (img_outputs.logits_per_image / 100).cpu().detach().numpy()
                )
                # print('img_scores.shape=', img_scores.shape)

                harmonic_mean = (
                    2 * text_scores * img_scores / (text_scores + img_scores)
                )
            # print('harmonic_mean=', harmonic_mean)
            return harmonic_mean
    elif eval_func == "clair":
        assert False, "clair is not available."
        # evaluator = GPT4Eval('gpt4')
        evaluator = GPT4Eval("gpt4mini")
        evaluator.set_prompt(_CLAIR_PROMPT)

        def compute_evaluate(hyp, ref, src):
            ref_list = "\n".join(ref)
            reward_value = evaluator.get_reward(ref_list, hyp)
            return reward_value
    elif eval_func == "sentbert":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        evaluator = AutoModel.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        evaluator.eval()

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def compute_evaluate(hyp, ref, src):
            with torch.no_grad():
                encoded_input = tokenizer(
                    [hyp, ref], padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                model_output = evaluator(**encoded_input)

                # Perform pooling
                sentence_embeddings = mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
                sentence_embeddings_norm = F.normalize(sentence_embeddings, p=2, dim=1)
                # print("sentence_embeddings_norm=", sentence_embeddings_norm)
                text_scores = (
                    cosine_similarity(
                        sentence_embeddings_norm[:1], sentence_embeddings_norm[1:]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .max()
                )
            return text_scores
    elif eval_func == "jreadability":
        evaluator = None

        def compute_evaluate(hyp, ref, src):
            scores = compute_readability(hyp)
            return scores
    elif eval_func == "parse-answer":
        evaluator = None

        def compute_evaluate(hyp, ref, src):
            try:
                answer = hyp.split("[[")[1].split("]]")[0]
                ans_num = int(answer)
                if ans_num == int(ref[0]):
                    reward_value = 1
                else:
                    reward_value = 0
            except Exception as e:
                ref = str(ref[0]) + ": " + str(ref[1])
                reward_value = (
                    "PARSE ERROR\n" + "CORRECT:" + ref + "\nSYSOUT: " + str(hyp)
                )
            return reward_value
    elif eval_func == "cer":
        evaluator = utility_class.CER()

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute_similarity(hyp=[hyp], ref=[ref], src=None)[0]
    elif eval_func == "cer-neologdn":
        evaluator = utility_class.CER(normalizer="neologdn")

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute_similarity(hyp=[hyp], ref=[ref], src=None)[0]
    elif eval_func == "cer-nonorm":
        evaluator = utility_class.CER(normalizer="none")

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute_similarity(hyp=[hyp], ref=[ref], src=None)[0]
    elif eval_func == "wer":
        evaluator = utility_class.WER()

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute_similarity(hyp=[hyp], ref=[ref], src=None)[0]
    elif eval_func == "wer-th":
        evaluator = utility_class.WER(normalizer="th")

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute_similarity(hyp=[hyp], ref=[ref], src=None)[0]
    elif eval_func == "wer-nonorm":
        evaluator = utility_class.WER(normalizer="none")

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute_similarity(hyp=[hyp], ref=[ref], src=None)[0]
    else:
        assert False

    return compute_evaluate, evaluator


def compute_self_score(hyps, src, compute_evaluate):
    scores = []
    n_samples = 0
    n = len(hyps)
    for i in range(n):
        for j in range(n):
            if i != j:
                score = compute_evaluate(hyps[i], hyps[j], src)
                scores.append(score)
                n_samples += 1
    return sum(scores) / n_samples


def distinct_n_diversity(sentences, n):
    """
    Compute distinct-N among a set of sentences.
    :param sentences: a list of sentences.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    assert n >= 1
    assert isinstance(sentences, list)
    if len(sentences) == 0:
        return 0.0  # Prevent a zero division
    assert isinstance(sentences[0], str)

    word_tokenizer = ToktokTokenizer()

    list_of_words = [word_tokenizer.tokenize(sentence) for sentence in sentences]

    distinct_ngrams = set()
    for words in list_of_words:
        # if len(words) == 0:
        #     continue
        if len(words) < n:
            continue
        d_ngrams = ngrams(words, n)
        distinct_ngrams.update(d_ngrams)

    if len(distinct_ngrams) == 0:
        return 0

    return len(distinct_ngrams) / sum([len(words) for words in list_of_words])


def evaluate_diversity(hyp, scores, src_input, compute_pairwise):
    """
    This function computes the metrics for the diversity experiments.
    kmbr_mean_score: mean score of the hypotheses.
    kmbr_min_score: min score of the hypotheses.
        -> These two metrics are used to compare the quality of the hypotheses.
    """
    if len(hyp) < 2:
        # If there is only one hypothesis, we cannot compute the diversity.
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # print('hyp=', hyp)
    kmbr_mean_score = sum(scores) / len(scores)
    kmbr_min_score = min(scores)
    kmbr_max_score = max(scores)
    kmbr_self_score = compute_self_score(hyp, src_input, compute_pairwise)
    kmbr_dn_1 = distinct_n_diversity(hyp, 1)
    kmbr_dn_2 = distinct_n_diversity(hyp, 2)
    kmbr_dn_3 = distinct_n_diversity(hyp, 3)
    return [
        kmbr_mean_score,
        kmbr_min_score,
        kmbr_max_score,
        kmbr_self_score,
        kmbr_dn_1,
        kmbr_dn_2,
        kmbr_dn_3,
    ]
