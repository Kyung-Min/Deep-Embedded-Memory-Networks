from __future__ import print_function

from collections import defaultdict
import numpy as np
import json
from operator import itemgetter

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

import re
from nltk.corpus import stopwords
from collections import namedtuple

from keras.callbacks import Callback

class Embedder(object):
    """ Generic embedding interface.

    Required: attributes g and N """

    def map_tokens(self, tokens, ndim=2):
        """ for the given list of tokens, return a list of GloVe embeddings,
        or a single plain bag-of-words average embedding if ndim=1.

        Unseen words (that's actually *very* rare) are mapped to 0-vectors. """
        gtokens = [self.g[t] for t in tokens if t in self.g]
        if not gtokens:
            return np.zeros((1, self.N)) if ndim == 2 else np.zeros(self.N)
        gtokens = np.array(gtokens)
        if ndim == 2:
            return gtokens
        else:
            return gtokens.mean(axis=0)

    def map_set(self, ss, ndim=2):
        """ apply map_tokens on a whole set of sentences """
        return [self.map_tokens(s, ndim=ndim) for s in ss]

    def pad_set(self, ss, spad, N=None):
        """ Given a set of sentences transformed to per-word embeddings
        (using glove.map_set()), convert them to a 3D matrix with fixed
        sentence sizes - padded or trimmed to spad embeddings per sentence.

        Output is a tensor of shape (len(ss), spad, N).

        To determine spad, use something like
            np.sort([np.shape(s) for s in s0], axis=0)[-1000]
        so that typically everything fits, but you don't go to absurd lengths
        to accomodate outliers.
        """
        ss2 = []
        if N is None:
            N = self.N
        for s in ss:
            if spad > s.shape[0]:
                if s.ndim == 2:
                    s = np.vstack((s, np.zeros((spad - s.shape[0], N))))
                else:  # pad non-embeddings (e.g. toklabels) too
                    s = np.hstack((s, np.zeros(spad - s.shape[0])))
            elif spad < s.shape[0]:
                s = s[:spad]
            ss2.append(s)
        return np.array(ss2)

class GloVe(Embedder):
    """ A GloVe dictionary and the associated N-dimensional vector space """
    def __init__(self, N=300, glovepath='glove.6B.%dd.txt'):
        """ Load GloVe dictionary from the standard distributed text file.

        Glovepath should contain %d, which is substituted for the embedding
        dimension N. """
        self.N = N
        self.g = dict()
        self.glovepath = glovepath % (N,)

        with open(self.glovepath, 'r') as f:
            for line in f:
                l = line.split()
                word = l[0]
                self.g[word] = np.array(l[1:]).astype(float)


def hash_params(pardict):
    ps = json.dumps(dict([(k, str(v)) for k, v in pardict.items()]), sort_keys=True)
    h = hash(ps)
    return ps, h


"""
NLP preprocessing tools for sentences.

Currently, this just tags the token sequences by some trivial boolean flags
that denote some token characteristics and sentence-sentence overlaps.

In principle, this module could however include a lot more sophisticated
NLP tagging pipelines, or loading precomputed such data.
"""

stop = stopwords.words('english')

flagsdim = 4

def sentence_flags(s0, s1, spad):
    """ For sentence lists s0, s1, generate numpy tensor
    (#sents, spad, flagsdim) that contains a sparse indicator vector of
    various token properties.  It is meant to be concatenated to the token
    embedding. """

    def gen_iflags(s, spad):
        iflags = []
        for i in range(len(s)):
            iiflags = [[False, False] for j in range(spad)]
            for j, t in enumerate(s[i]):
                if j >= spad:
                    break
                number = False
                capital = False
                if re.match('^[0-9\W]*[0-9]+[0-9\W]*$', t):
                    number = True
                if j > 0 and re.match('^[A-Z]', t):
                    capital = True
                iiflags[j] = [number, capital]
            iflags.append(iiflags)
        return iflags

    def gen_mflags(s0, s1, spad):
        """ generate flags for s0 that represent overlaps with s1 """
        mflags = []
        for i in range(len(s0)):
            mmflags = [[False, False] for j in range(spad)]
            for j in range(min(spad, len(s0[i]))):
                unigram = False
                bigram = False
                for k in range(len(s1[i])):
                    if s0[i][j].lower() != s1[i][k].lower():
                        continue
                    # do not generate trivial overlap flags, but accept them as part of bigrams                    
                    if s0[i][j].lower() not in stop and not re.match('^\W+$', s0[i][j]):
                        unigram = True
                    try:
                        if s0[i][j+1].lower() == s1[i][k+1].lower():
                            bigram = True
                    except IndexError:
                        pass
                mmflags[j] = [unigram, bigram]
            mflags.append(mmflags)
        return mflags

    # individual flags (for understanding)
    iflags0 = gen_iflags(s0, spad)
    iflags1 = gen_iflags(s1, spad)

    # s1-s0 match flags (for attention)
    mflags0 = gen_mflags(s0, s1, spad)
    mflags1 = gen_mflags(s1, s0, spad)

    return [np.dstack((iflags0, mflags0)),
            np.dstack((iflags1, mflags1))]



"""
Vocabulary that indexes words, can handle OOV words and integrates word
embeddings.
"""

class Vocabulary:
    """ word-to-index mapping, token sequence mapping tools and
    embedding matrix construction tools """
    def __init__(self, sentences, count_thres=1):
        """ build a vocabulary from given list of sentences, but including
        only words occuring at least #count_thres times """

        # Counter() is superslow :(
        vocabset = defaultdict(int)
        for s in sentences:
            for t in s:
                vocabset[t] += 1

        vocab = sorted(list(map(itemgetter(0),
                                filter(lambda k: itemgetter(1)(k) >= count_thres,
                                       vocabset.items() ) )))
        self.word_idx = dict((w, i + 2) for i, w in enumerate(vocab))
        self.word_idx['_PAD_'] = 0
        self.word_idx['_OOV_'] = 1
        print('Vocabulary of %d words' % (len(self.word_idx)))

        self.embcache = dict()

    def add_word(self, word):
        if word not in self.word_idx:
            self.word_idx[word] = len(self.word_idx)

    def vectorize(self, slist, pad=60):
        """ build an pad-ed matrix of word indices from a list of
        token sequences """
        silist = [[self.word_idx.get(t, 1) for t in s] for s in slist]
        if pad is not None:
            return pad_sequences(silist, maxlen=pad, truncating='post', padding='post') 
        else:
            return silist

    def embmatrix(self, emb):
        """ generate index-based embedding matrix from embedding class emb
        (typically GloVe); pass as weights= argument of Keras' Embedding layer """
        if str(emb) in self.embcache:
            return self.embcache[str(emb)]
        embedding_weights = np.zeros((len(self.word_idx), emb.N))
        for word, index in self.word_idx.items():
            try:
                embedding_weights[index, :] = emb.g[word]
            except KeyError:
                if index == 0:
                    embedding_weights[index, :] = np.zeros(emb.N)
                else:
                    embedding_weights[index, :] = np.random.uniform(-0.25, 0.25, emb.N)  # 0.25 is embedding SD
        self.embcache[str(emb)] = embedding_weights
        return embedding_weights

    def size(self):
        return len(self.word_idx)

"""
Evaluation tools, mainly non-straightforward methods.
"""

def aggregate_q(q, y, ypred):
    """
    Generate tuples (q, [(y, ypred), ...]) where the list is sorted
    by the ypred score.  
    """
    ybyq = dict()
    for i in range(len(q)):
        try:
            qis = q[i].tostring()
        except AttributeError:
            qis = str(q[i])
        if qis in ybyq:
            ybyq[qis].append((y[i], ypred[i], q[i], i))
        else:
            ybyq[qis] = [(y[i], ypred[i], q[i], i)]

    for s, yl in ybyq.items():
        ys = sorted(yl, key=lambda yy: yy[1], reverse=True)
        yield (s, ys)

def mrr(q, y, ypred):
    """
    Compute MRR (mean reciprocial rank) of y-predictions, by grouping
    y-predictions for the same q together.  This metric is relevant
    e.g. for the "answer sentence selection" task where we want to
    identify and take top N most relevant sentences.
    """
    rr = []
    best_s_idxes = []
    for s, ys in aggregate_q(q, y, ypred):
        if np.sum([yy[0] for yy in ys]) == 0:
            continue  # do not include q with no right answer sentences in MRR
        ysd = dict()
        isd = dict()
        for yy in ys:
            score = yy[1][0]
            if score in ysd:
                ysd[score].append(yy[0])
                isd[score].append(yy[3])
            else:
                ysd[score] = [yy[0]]
                isd[score] = [yy[3]]
        rank = 0
        for yp in sorted(ysd.keys(), reverse=True):
            if np.sum(ysd[yp]) > 0:
                rankofs = 1 - np.sum(ysd[yp]) / len(ysd[yp])
                rank += (len(ysd[yp])/4) * rankofs
                break
            rank += len(ysd[yp])/4
        rr.append(1 / float(1+rank))

        best_s_idxes.append([isd.values()[0]])
       
    return (np.mean(rr), best_s_idxes) 

def acc(best_s_idxes, ypredA1, ypredA2):
    n_true = 0
    for idxes in best_s_idxes:
        scores = []
        idxes = idxes[0]
        array_tmp = []
        for idx in idxes:
            scores.append(ypredA2[idx])
        scores.append(ypredA1[idxes[0]])
        if np.argmax(scores) == len(idxes):
            n_true += 1
        
    return float(n_true)/float(len(best_s_idxes)) 

AnsSelRes = namedtuple('AnsSelRes', ['MRR', 'MAP'])

def eval_QA(ypredS, ypredA1, ypredA2, q, y, MAP=False):
    mrr_, best_s_idxes= mrr(q, y, ypredS)
    acc_ = acc(best_s_idxes, ypredA1, ypredA2)

    print('MRR: %f' % (mrr_))
    print('Accuracy: %f' %(acc_))

    map_ = '_'
    return AnsSelRes(mrr_, map_)


"""
Task-specific callbacks for the fit() function.
"""

class AnsSelCB(Callback):
    """ A callback that monitors answer selection validation ACC after each epoch """
    def __init__(self, val_q, val_s1, val_s2, val_q_s1, val_a1, val_a2, y, inputs):
        self.val_q = val_q
        self.val_s1 = val_s1
        self.val_s2 = val_s2
        self.val_q_s1 = val_q_s1
        self.val_a1 = val_a1
        self.val_a2 = val_a2
        self.val_y = y 
        self.val_inputs = inputs

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.val_inputs)
        ypredS = pred[0]
        ypredA1 = pred[2]
        ypredA2 = pred[3]
        mrr_, best_s_idxes = mrr(self.val_q, self.val_y, ypredS)
        print('val MRR %f' % (mrr_,))
        logs['mrr'] = mrr_
        acc_ = acc(best_s_idxes, ypredA1, ypredA2)
        print('val ACC %f' % (acc_,))
        logs['acc'] = acc_
