from collections import Counter

import numpy as np
import spacy

from utility.utility_class import UtilityFunction


class DSARI(UtilityFunction):
    def __init__(self, ngrams=4):
        self.ngrams = ngrams
        self.nlp = spacy.load("ja_ginza")
        self.nlp.add_pipe("sentencizer")

    def compute_similarity(self, hyp, ref, src):
        return self.dsari(hyp, ref, src, self.ngrams)

    def dsari(self, hyp, ref, src, ngrams=4):
        # The utility function requires source sentence.
        assert isinstance(src, str)
        assert isinstance(hyp, str)
        assert isinstance(ref, str)
        assert isinstance(ngrams, int)

        i_tokens = self.text2tokens(src)
        o_tokens = self.text2tokens(hyp)
        r_tokens = self.text2tokens(ref)
        # i_sents = text2sentences(src)
        o_sents = self.text2sentences(hyp)
        r_sents = self.text2sentences(ref)
        I = len(i_tokens)
        O = len(o_tokens)
        R = len(r_tokens)
        Os = len(o_sents)
        Rs = len(r_sents)
        i_ngrams = [self.get_ngrams(i_tokens, n) for n in range(1, ngrams + 1)]
        o_ngrams = [self.get_ngrams(o_tokens, n) for n in range(1, ngrams + 1)]
        r_ngrams = [self.get_ngrams(r_tokens, n) for n in range(1, ngrams + 1)]

        Fkeep, Fadd, Pdel = self.sari(i_ngrams, o_ngrams, r_ngrams)
        # print("Fkeep, Fadd, Pdel=", Fkeep, Fadd, Pdel)
        lp1 = self.LP1(I, O, R)
        lp2 = self.LP2(I, O, R)
        slp = self.SLP(Os, Rs)
        # print("lp1, lp2, slp=", lp1, lp2, slp)
        return (Fkeep * lp2 * slp + Fadd * lp1 + Pdel * lp2) / 3.0

    def text2tokens(self, text):
        with self.nlp.select_pipes(enable=["parser"]):
            doc = self.nlp(text)
        tokens = [s.text for s in doc]
        return tokens

    def text2sentences(self, text):
        with self.nlp.select_pipes(enable=["sentencizer"]):
            doc = self.nlp(text)
        sentences = [s.text for s in doc.sents]
        return sentences

    def get_ngrams(self, tokens, ngram_range):
        return Counter(
            [
                " ".join(tokens[i : i + ngram_range])
                for i in range(len(tokens) - ngram_range + 1)
            ]
        )

    # Definition of the F values are from the following paper
    # https://aclanthology.org/Q16-1029.pdf
    def prkeep(self, i_ngram, o_ngram, r_ngram):
        assert isinstance(i_ngram, Counter)

        i_and_o = i_ngram & o_ngram
        i_and_r = i_ngram & r_ngram

        ioir = 0
        for voc in o_ngram.keys():
            # Note: If voc is not in the dict, Counter returns 0.
            ioir += min(i_and_o[voc], i_and_r[voc])

        io = i_and_o.total()
        ir = i_and_r.total()

        if io == 0:
            pkeep = 0
        else:
            pkeep = ioir / io
        if ir == 0:
            rkeep = 0
        else:
            rkeep = ioir / ir
        assert pkeep <= 1
        assert rkeep <= 1
        return pkeep, rkeep

    def pradd(self, i_ngram, o_ngram, r_ngram):
        assert isinstance(i_ngram, Counter)

        o_min_i = o_ngram - i_ngram
        r_min_i = r_ngram - i_ngram
        oir = 0
        for voc in o_ngram.keys():
            oir += min(o_min_i[voc], r_ngram[voc])

        oi = o_min_i.total()
        ri = r_min_i.total()

        if oi == 0:
            padd = 0
        else:
            padd = oir / oi
        if ri == 0:
            radd = 0
        else:
            radd = oir / ri
        assert padd <= 1
        assert radd <= 1
        return padd, radd

    def pdel(self, i_ngram, o_ngram, r_ngram):
        assert isinstance(i_ngram, Counter)

        i_min_o = i_ngram - o_ngram
        i_min_r = i_ngram - r_ngram
        # print('i_ngram', i_ngram)
        # print('o_ngram', o_ngram)
        # print('r_ngram', r_ngram)
        # print('i_min_o', i_min_o)
        # print('i_min_r', i_min_r)
        ior = 0
        for voc in i_ngram.keys():
            ior += min(i_min_o[voc], i_min_r[voc])

        io = i_min_o.total()
        if io == 0:
            pdel = 0
        else:
            pdel = ior / io
        # print(ior, io)
        assert pdel <= 1
        return pdel

    def sari(self, i_ngrams, o_ngrams, r_ngrams):
        assert len(i_ngrams) == len(o_ngrams) == len(r_ngrams)
        n_grams = len(i_ngrams)

        Pkeep = 0
        Rkeep = 0
        Padd = 0
        Radd = 0
        Pdel = 0
        for n in range(n_grams):
            i_ngram = i_ngrams[n]
            o_ngram = o_ngrams[n]
            r_ngram = r_ngrams[n]

            pkeep, rkeep = self.prkeep(i_ngram, o_ngram, r_ngram)
            padd, radd = self.pradd(i_ngram, o_ngram, r_ngram)
            pdel_ = self.pdel(i_ngram, o_ngram, r_ngram)
            Pkeep += pkeep
            Rkeep += rkeep
            Padd += padd
            Radd += radd
            Pdel += pdel_
        Pkeep /= n_grams
        Rkeep /= n_grams
        Padd /= n_grams
        Radd /= n_grams
        Pdel /= n_grams

        assert Pkeep <= 1
        assert Rkeep <= 1
        assert Padd <= 1
        assert Radd <= 1
        assert Pdel <= 1

        if Pkeep + Rkeep == 0:
            Fkeep = 0
        else:
            Fkeep = 2.0 * Pkeep * Rkeep / (Pkeep + Rkeep)
        if Padd + Radd == 0:
            Fadd = 0
        else:
            Fadd = 2.0 * Padd * Radd / (Padd + Radd)
        # sari = (Fkeep + Fadd + Pdel) / 3.0
        assert Fkeep <= 1
        assert Fadd <= 1
        assert Pdel <= 1
        return Fkeep, Fadd, Pdel

    def LP1(self, i, o, r):
        assert isinstance(i, int)
        assert isinstance(o, int)
        assert isinstance(r, int)
        if o >= r:
            return 1.0
        else:
            return np.exp((o - r) / o)

    def LP2(self, i, o, r):
        assert isinstance(i, int)
        assert isinstance(o, int)
        assert isinstance(r, int)
        if o <= r:
            return 1.0
        else:
            return np.exp((r - o) / max(i - r, 1))

    def SLP(self, os, rs):
        assert isinstance(os, int)
        assert isinstance(rs, int)
        return np.exp(-np.abs(rs - os) / max(rs, os))
