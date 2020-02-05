
import bernmix as bm
import numpy as np


class LinModel:
    def __init__(self, weights=None, snp_names=None, probs=None):
        """
        Initialisation of linear model
        :param weights: loadings
        :param snp_names: names of SNPs
        :param probs: probabilities of reference allele in population
        """
        self.weights = list(weights)
        self.snps = snp_names
        self.probs = probs
        self.check_probs()

    def check_probs(self):
        if not all(0 <= p <= 1 for p in self.probs):
            raise ValueError('Probabilities are not in [0 1] range')

    def load_dataset(self, q_data, n_data=None):
        """
        Definition of probabilities
        :param q_data: Amounts of reference allele within each loci
        :param n_data: Number of individuals summarised in the loci
        """

        if not all(snp in q_data.columns for snp in self.snps):
            raise IndexError('not ALL SNPs are in the dataset: {}'.format(set(snps) - set(q_data.columns)))

        q_tmp = q_data[self.snps]

        if n_data is None:
            self.probs = q_tmp.sum() / 2
            self.check_probs()
            return

        if not all(snp in n_data.columns for snp in self.snps):
            raise IndexError('not ALL SNPs are in the dataset: {}'.format(set(snps) - set(n_data.columns)))

        n_tmp = n_data[self.snps]
        self.probs = q_tmp.sum() / n_tmp.sum()
        self.check_probs()

    def estimate_probs(self):
        """
        Estimate probabilities of specific values
        :return:
        """
        # to enlarge the number of elements in linear combination of BRVs
        weights = self.weights * 2
        probs = list(self.probs) * 2

        # Scaling weights
        scale = 1
        min_abs_w = min([abs(w) for w in weights])
        if min_abs_w < 1:
            scale = 1 / min_abs_w

        scale *= 10 ** 4
        weights = [round(w * scale) for w in weights]

        pmf, values = bm.pmf_int_vals(probs, weights)

        pthresh = 10 ** -7
        pmf = [0 if p < pthresh else p for p in pmf]
        pscale = 1 / sum(pmf)
        pmf = [p * pscale for p in pmf]

        pmf, values = zip(*[(pmf[i], values[i]/scale) for i in range(len(pmf)) if pmf[i] > 0])

        return pmf, values


class TwoPhenotypes(LinModel):
    def __init__(self, lm1: LinModel, lm2: LinModel):
        super(LinModel, self).__init__()
        self.snps = list(set(lm1.snps) | set(lm2.snps))

        self.probs, self.w1, self.w2 = self.define_params(lm1, lm2)


    def transform_params(self, lm: LinModel):
        """
        Get probabilities and weights corresponding to current model
        :param lm: LinModel-class object
        :return: probs and weights
        """
        w = [lm.weights[lm.snps.index(snp)] if snp in lm.snps else 0
             for snp in self.snps]
        p = [lm.probs[lm.snps.index(snp)] if snp in lm.snps else 0
             for snp in self.snps]
        return p, w

    def define_params(self, lm1: LinModel, lm2: LinModel):
        """
        Define probabilities and weights for two models
        :param lm1:
        :param lm2:
        :return:
        """
        p1, w1 = self.transform_params(lm1)
        p2, w2 = self.transform_params(lm2)

        for i in range(len(self.snps)):
            if (w1[i] != 0) and (w2[i] != 0) and (p1[i] != p2[i]):
                raise ValueError('Linear Models are not consistent')

        p = [max(p1_elem, p2_elem) for p1_elem, p2_elem in zip(p1, p2)]

        return p, w1, w2

    def estimate_distr(self):
        """
        Estimate of the joint distribution of two phenotypes
        :return: PMF, and two coordinates
        """

        w1_range = sum([w if w > 0 else -w for w in self.w1])

        self.weights = [self.w1[i] + w1_range * self.w2[i]
                        for i in range(len(self.probs))]

        pmf, values = self.estimate_probs()

        sign_bias = min(values)
        values = [v - sign_bias for v in values]

        val1, val2 = self.values2pairs(values, w1_range)
        sign_bias1 = [w for w in self.w1 if w < 0]
        sign_bias2 = [w for w in self.w2 if w < 0]

        val1 = [v + sign_bias1 for v in val1]
        val2 = [v + sign_bias2 for v in val2]

        return pmf, val1, val2

    def values2pairs(self, values, range):
        """
        Transform values to 2 components
        """
        val1 = [ v % range for v in values]
        val2 = [(v-v1) / range  for v, v1 in zip(values, val1)]
        return val1, val2





















