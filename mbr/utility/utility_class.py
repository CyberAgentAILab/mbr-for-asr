from collections.abc import Iterable

import numpy as np
from nltk.tokenize import ToktokTokenizer
import MeCab
import neologdn
import spacy_thai

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from transformers import (
    CLIPTextModel,
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoTokenizer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from evaluate import load


class UtilityFunction:
    def __init__(self):
        self.similarity = None

    def compute_similarity(self, hyp: Iterable, ref: Iterable, src):
        pass

    def compute_score_matrix(self, samples, src=None):
        n_samples = len(samples)
        scores = []
        for i in range(n_samples):
            score = self.compute_similarity(
                hyp=np.array([samples[i]] * n_samples), ref=samples, src=src
            )
            scores.append(score)
        return np.array(scores)


class BLEURT(UtilityFunction):
    def __init__(self):
        self.similarity = load("bleurt", checkpoint="BLEURT-20")

    def compute_similarity(self, hyp, ref, src):
        return self.similarity.compute(predictions=hyp, references=ref)["scores"]


class COMET(UtilityFunction):
    def __init__(self):
        from comet import download_model, load_from_checkpoint

        self.similarity = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    def compute_similarity(self, hyp, ref, src):
        data = []
        for i in range(len(hyp)):
            d = {}
            d["src"] = src
            d["mt"] = hyp[i]
            d["ref"] = ref[i]
            data.append(d)
        model_output = self.similarity.predict(data, batch_size=128)
        return model_output.scores

    def compute_embedding(self, hyp):
        with torch.no_grad():
            batch = self.similarity.encoder.prepare_sample(hyp).to(
                self.similarity.device
            )
            emb = self.similarity.get_sentence_embedding(**batch).cpu().detach().numpy()
        return emb


class COMET20(COMET):
    def __init__(self):
        from comet import download_model, load_from_checkpoint

        self.similarity = load_from_checkpoint(download_model("Unbabel/wmt20-comet-da"))


class BERTSCORE(UtilityFunction):
    def __init__(self):
        self.similarity = load("bertscore")

    def compute_similarity(self, hyp, ref, src):
        return self.similarity.compute(predictions=hyp, references=ref, lang="en")["f1"]


class DEBERTA(BERTSCORE):
    def __init__(self):
        self.similarity = load("bertscore")

    def compute_similarity(self, hyp, ref, src):
        return self.similarity.compute(
            predictions=hyp,
            references=ref,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
        )["f1"]


class SACREBLEU(UtilityFunction):
    def __init__(self, language="en"):
        self.similarity = load("sacrebleu")
        self.language = language

        # if self.language == 'ja':
        #     self.tokenizer = MeCabTokenizer()

    def compute_similarity(self, hyp, ref, src):
        if self.language == "en":
            scores = [
                self.similarity.compute(predictions=[hyp[i]], references=[ref[i]])[
                    "score"
                ]
                / 100.0
                for i in range(len(hyp))
            ]
        elif self.language == "ja":
            scores = [
                self.similarity.compute(
                    predictions=[hyp[i]], references=[ref[i]], tokenize="ja-mecab"
                )["score"]
                / 100.0
                for i in range(len(hyp))
            ]
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        return scores


class ROUGEL(UtilityFunction):
    def __init__(self):
        self.similarity = load("rouge")

    def compute_similarity(self, hyp, ref, src):
        scores = [
            self.similarity.compute(predictions=[hyp[i]], references=[ref[i]])["rougeL"]
            for i in range(len(hyp))
        ]
        return scores


class INFOLM(UtilityFunction):
    def __init__(self):
        self.similarity = InfoLM(
            "google/bert_uncased_L-2_H-128_A-2",
            information_measure="fisher_rao_distance",
            idf=False,
            return_sentence_level_score=True,
        )

    def compute_similarity(self, hyp, ref, src):
        return -np.array(self.similarity(hyp, ref)[1])


class CLIP(UtilityFunction):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/clip-vit-large-patch14"
        # model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(model_id)

        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.similarity = CLIPTextModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.similarity.eval()

    def compute_similarity(self, hyp, ref, src):
        with torch.no_grad():
            hyp = list(hyp)
            ref = list(ref)
            inputs = self.processor(
                text=hyp + ref,
                images=src[0],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            text_embeddings = torch.flatten(
                self.similarity(inputs.input_ids.to(self.device))["last_hidden_state"],
                1,
                -1,
            )
            hyp_embeddings = text_embeddings[: len(hyp)]
            ref_embeddings = text_embeddings[len(hyp) :]
            text_scores = (
                cosine_similarity(hyp_embeddings, ref_embeddings).cpu().detach().numpy()
            )

            img_inputs = self.processor(
                text=hyp,
                images=src[0],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            img_outputs = self.model(**img_inputs)

            img_scores = np.squeeze(
                (img_outputs.logits_per_image / 100).cpu().detach().numpy()
            )
            # print('img_scores.shape=', img_scores.shape)

            harmonic_mean = 2 * text_scores * img_scores / (text_scores + img_scores)
        # print('harmonic_mean=', harmonic_mean)
        return harmonic_mean


class CLIPTEXT(UtilityFunction):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/clip-vit-large-patch14"
        self.processor = CLIPProcessor.from_pretrained(model_id)

        self.similarity = CLIPTextModel.from_pretrained(model_id).to(self.device)
        self.similarity.eval()

    def compute_similarity(self, hyp, ref, src):
        with torch.no_grad():
            hyp = list(hyp)
            ref = list(ref)
            inputs = self.processor(
                text=hyp + ref,
                images=src,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            text_embeddings = torch.flatten(
                self.similarity(inputs.input_ids.to(self.device))["last_hidden_state"],
                1,
                -1,
            )
            hyp_embeddings = text_embeddings[: len(hyp)]
            ref_embeddings = text_embeddings[len(hyp) :]
            text_scores = (
                cosine_similarity(hyp_embeddings, ref_embeddings).cpu().detach().numpy()
            )

        return text_scores

    def compute_mean_embedding_scores(self, samples, src=None):
        with torch.no_grad():
            hyps = list(samples)
            inputs = self.processor(
                text=hyps,
                images=src,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            text_embeddings = torch.flatten(
                self.similarity(inputs.input_ids.to(self.device))["last_hidden_state"],
                1,
                -1,
            )

            mean_embedding = torch.mean(text_embeddings, dim=0)

            text_scores = []
            for i in range(len(hyps)):
                score = (
                    cosine_similarity(
                        text_embeddings[i : i + 1], torch.unsqueeze(mean_embedding, 0)
                    )
                    .cpu()
                    .detach()
                    .numpy()[0]
                )
                # print('CLIPTEXT score=', score)
                text_scores.append(score)

        return text_scores


class UNIGRAMF1(UtilityFunction):
    def __init__(self):
        self.similarity = ToktokTokenizer()

    def compute_similarity(self, hyp, ref, src):
        nhyp = len(hyp)
        f1s = []
        for i in range(nhyp):
            h = hyp[i]
            r = ref[i]
            hyp_tok = self.similarity.tokenize(h)
            ref_tok = self.similarity.tokenize(r)

            if len(hyp_tok) == 0 or len(ref_tok) == 0:
                f1s.append(0.0)
            else:
                precision = len([token for token in hyp_tok if token in ref_tok]) / len(
                    hyp_tok
                )
                recall = len([token for token in hyp_tok if token in ref_tok]) / len(
                    ref_tok
                )

                if precision + recall < 0.0001:
                    # Prevent zero division.
                    f1s.append(0.0)
                else:
                    f1s.append(2.0 * precision * recall / (precision + recall))
        return f1s


class SENTBERT(UtilityFunction):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.evaluator = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.evaluator.eval()
        self.evaluator.to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def compute_embedding(self, hyp):
        with torch.no_grad():
            encoded_input = self.tokenizer(
                hyp, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.evaluator(**encoded_input)

            # Perform pooling
            sentence_embeddings = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            sentence_embeddings_norm = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings_norm

    def compute_similarity(self, hyp, ref, src):
        hyp = list(hyp)
        ref = list(ref)
        # print('hyp=', hyp)
        # print('ref=', ref)
        sentence_embeddings_norm = self.compute_embedding(hyp + ref)

        text_scores = []
        for i in range(len(hyp)):
            text_score = (
                cosine_similarity(
                    sentence_embeddings_norm[i : i + 1],
                    sentence_embeddings_norm[len(hyp) + i : len(hyp) + i + 1],
                )
                .cpu()
                .detach()
                .numpy()
                .max()
            )
            text_scores.append(text_score)
        return text_scores

    def compute_score_matrix(self, samples, src=None):
        sentence_embeddings_norm = self.compute_embedding(list(samples))
        n_samples = len(samples)

        score_matrix = np.zeros([n_samples, n_samples])

        for i in range(n_samples):
            for j in range(i, n_samples):
                score = (
                    cosine_similarity(
                        sentence_embeddings_norm[i : i + 1],
                        sentence_embeddings_norm[j : j + 1],
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .max()
                )
                # print('SENTBERT score=', score)
                score_matrix[i, j] = score
                score_matrix[j, i] = score

        # print('SENTBERT score_matrix=', score_matrix)
        return np.array(score_matrix)

    def compute_mean_embedding_scores(self, samples, src=None):
        sentence_embeddings_norm = self.compute_embedding(list(samples))
        n_samples = len(samples)

        mean_embedding = torch.mean(sentence_embeddings_norm, dim=0)

        score_list = np.zeros([n_samples])

        for i in range(n_samples):
            score = (
                cosine_similarity(
                    sentence_embeddings_norm[i : i + 1],
                    torch.unsqueeze(mean_embedding, 0),
                )
                .cpu()
                .detach()
                .numpy()
                .max()
            )
            # print('SENTBERT score=', score)
            score_list[i] = score

        return score_list


class SFR(UtilityFunction):
    def __init__(self, model_id="Salesforce/SFR-Embedding-2_R"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.evaluator = AutoModel.from_pretrained(model_id, device_map="auto")

        self.evaluator.eval()
        # self.evaluator.to(self.device)

        self.instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

    def compute_embedding(self, hyp, ref):
        max_length = 4096
        queries = [self.get_detailed_instruct(self.instruction, h) for h in hyp]
        passages = ref
        input_texts = queries + passages
        # print('input_texts=', input_texts)
        batch_dict = self.tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = self.evaluator(**batch_dict)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def compute_similarity(self, hyp, ref, src):
        hyp = list(hyp)
        ref = list(ref)

        assert len(hyp) == len(ref)

        scores = []
        for i in range(len(hyp)):
            hi = [hyp[i]]
            ri = [ref[i]]

            embeddings = self.compute_embedding(hi, ri)
            score = (
                cosine_similarity(embeddings[:1], embeddings[1:])
                .cpu()
                .detach()
                .numpy()[0]
            )
            scores.append(score)
        scores = np.array(scores)
        return scores

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]


class CER(UtilityFunction):
    def __init__(self, normalizer="default"):
        self.similarity = load("cer")

        if normalizer == "default":
            normalizer_ = BasicTextNormalizer()
            self.normalizer = (
                lambda x: normalizer_(x).replace(" ", "").replace("。.", "。")
            )
        elif normalizer == "neologdn":
            self.normalizer = neologdn.normalize
        elif normalizer == "none":
            self.normalizer = lambda x: x
        else:
            raise ValueError(f"Unknown normalizer: {normalizer}")

    def normalize(self, text):
        return self.normalizer(str(text))

    def compute_similarity(self, hyp, ref, src):
        # src is not used for this function.
        hyp = list(hyp)
        ref = list(ref)

        # Apply normalization using neologdn
        hyp_n = [self.normalize(h) for h in hyp]
        ref_n = [self.normalize(r) for r in ref]

        for i, r in enumerate(ref_n):
            if len(r) == 0:
                print("String empty in normalized reference:", r)
                print("Original reference:", ref[i])

        # Compute CER for each pair of hypothesis and reference
        scores = [
            1.0 - self.similarity.compute(predictions=[hyp_n[i]], references=[ref_n[i]])
            for i in range(len(hyp))
        ]
        return scores


class WER(UtilityFunction):
    def __init__(self, normalizer="default"):
        self.similarity = load("wer")

        if normalizer == "default":
            normalizer_ = BasicTextNormalizer()
            self.normalizer = lambda x: normalizer_(x)
        elif normalizer == "th":
            # Thai normalization
            self.nlp = spacy_thai.load()
            self.normalizer = lambda x: " ".join([token.text for token in self.nlp(x)])
        elif normalizer == "none":
            self.normalizer = lambda x: x
        else:
            raise ValueError(f"Unknown normalizer: {normalizer}")

    def normalize(self, text):
        return self.normalizer(str(text))

    def compute_similarity(self, hyp, ref, src):
        # src is not used for this function.
        hyp = list(hyp)
        ref = list(ref)
        # Apply normalization using neologdn
        hyp = [self.normalize(h) for h in hyp]
        ref = [self.normalize(r) for r in ref]
        # Compute CER for each pair of hypothesis and reference
        scores = [
            1.0 - self.similarity.compute(predictions=[hyp[i]], references=[ref[i]])
            for i in range(len(hyp))
        ]
        return scores


class MeCabTokenizer:
    def __init__(self, use_stemmer=False):
        self._stemmer = use_stemmer

        self.tagger = MeCab.Tagger()
        self.wakati = MeCab.Tagger("-Owakati")

    def tokenize(self, text):
        if self._stemmer:
            node = self.tagger.parseToNode(text)
            original_forms = []
            while node:
                feature = node.feature.split(",")
                original_forms.append(feature[6])
                node = node.next

            return original_forms

        else:
            return self.wakati.parse(text).split()
