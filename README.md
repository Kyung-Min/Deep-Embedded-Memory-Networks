# Deep-Embedded-Memory-Networks (Keras version)

##### Authors: Kyung-Min Kim, Min-Oh Heo, Seong-Ho Choi, and Byoung-Tak Zhang (Seoul National University & Surromind Robotics)
##### Paper: DeepStory: Video Story QA by Deep Embedded Memory Networks (https://arxiv.org/abs/1707.00836) (IJCAI 2017)

This notebook shows how the DEMN works. The DEMN consists of three modules (video story understanding, story selection, answer selection). This code corresponds to QA modules (story selection, answer selection) among them. The results of the video understanding module are reflected in the data provided (We are solving the copyright problem of the data. Soon the data will be released).

```python
from __future__ import print_function
from __future__ import division

import numpy as np
import sys

import utils

import keras.activations as activations
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, TimeDistributed
from keras.layers.merge import concatenate, add, multiply
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda, Permute, RepeatVector
from keras.layers.recurrent import GRU, LSTM

from keras import backend as K

import csv

def config():
    c = dict()
    # embedding params
    c['emb'] = 'Glove'
    c['embdim'] = 300
    c['inp_e_dropout'] = 1/2

    # objective function
    c['loss'] = 'ranking_loss'  
    c['margin'] = 1

    # training hyperparams
    c['opt'] = 'adam'
    c['batch_size'] = 160   
    c['epochs'] = 16
    
    # sentences with word lengths below the 'pad' will be padded with 0.
    c['pad'] = 60
    
    # scoring function: word-level attention-based model
    c['dropout'] = 1/2     
    c['dropoutfix_inp'] = 0
    c['dropoutfix_rec'] = 0           
    c['l2reg'] = 1e-4
                                              
    c['rnnbidi'] = True                      
    c['rnn'] = GRU                                                     
    c['rnnbidi_mode'] = add
    c['rnnact'] = 'tanh'
    c['rnninit'] = 'glorot_uniform'                      
    c['sdim'] = 1

    c['pool_layer'] = MaxPooling1D
    c['cnnact'] = 'tanh'
    c['cnninit'] = 'glorot_uniform'
    c['cdim'] = 2
    c['cfiltlen'] = 3
    
    c['adim'] = 1/2

    # mlp scoring function
    c['Ddim'] = 2
    
    ps, h = utils.hash_params(c)

    return c, ps, h
    
    
conf = None
emb = None
vocab = None
inp_tr = None
inp_val = None
inp_test = None
y_val = None
y_test = None

```

## Data Load

The data provided contain the output from the video story understanding module, i.e. reconstructed story sentence ![first eq](https://latex.codecogs.com/gif.latex?%24s_i%24), where 

![second eq](https://latex.codecogs.com/gif.latex?s_i%3De_i%7C%7Cl_i)
- ![third eq](https://latex.codecogs.com/gif.latex?%24e_i%24) is the description for the i-th video scene, which is retrieved by the video story understanding module <br>
- ![fourth eq](https://latex.codecogs.com/gif.latex?%24l_i%24) is the subtitle of the i-th video scene <br>
- || is concatenation <br>

For example, ![first eq](https://latex.codecogs.com/gif.latex?%24s_i%24) can be ‘there are three friends on the ground. the friends are talking about the new house.’


```python
'''
The format of the dataset is as follows.

Training dataset:
question1, positive story sentence, negative story sentence1, positive answer sentence, negative answer sentence1
                                        ...                                           , negative answer sentence2
                                        ...                                           , negative answer sentence3
                                        ...                                           , negative answer sentence4
question1, positive story sentence, negative story sentence2, positive answer sentence, negative answer sentence1
                                        ...                                           , negative answer sentence2
                                        ...                                           , negative answer sentence3
                                        ...                                           , negative answer sentence4
                                         
                                         ...
                                         
question2, positive story sentence, negative story sentence1, positive answer sentence, negative answer sentence1
                                        ...                                           , negative answer sentence2
                                        ...                                           , negative answer sentence3
                                        ...                                           , negative answer sentence4
question2, positive story sentence, negative story sentence2, positive answer sentence, negative answer sentence1
                                        ...                                           , negative answer sentence2
                                        ...                                           , negative answer sentence3
                                        ...                                           , negative answer sentence4

Validation & test dataset:
question1, label for story sentence, story sentence, dummy, positive answer sentence, negative answer sentence1
                                      ...                                           , negative answer sentence2
                                      ...                                           , negative answer sentence3
                                      ...                                           , negative answer sentence4
question1, label for story sentence, story sentence, dummy, positive answer sentence, negative answer sentence1
                                      ...                                           , negative answer sentence2
                                      ...                                           , negative answer sentence3
                                      ...                                           , negative answer sentence4
                                         
                                      ...
                                         
question2, label for story sentence, story sentence, dummy, positive answer sentence, negative answer sentence1
                                      ...                                           , negative answer sentence2
                                      ...                                           , negative answer sentence3
                                      ...                                           , negative answer sentence4
question2, label for story sentence, story sentence, dummy, positive answer sentence, negative answer sentence1
                                      ...                                           , negative answer sentence2
                                      ...                                           , negative answer sentence3
                                      ...                                           , negative answer sentence4
'''

def load_data_from_file(dsfile, iseval):
    #load a dataset in the csv format;

    q = [] # a set of questions
    s_p = [] # if training time, s1 is a set of positive sentences. Otherwise, s1 is a set of sentences.
    s_n = [] # if training time, s2 is a set of negative sentences. Otherwise, s2 is a set of dummy sentences.
    q_sp = [] # a set of sentences which concatenate questions and positive sentences
    a_p = [] # a set of positive answers
    a_n = [] # a set of negative answers
    labels = []

    with open(dsfile) as f:
        c = csv.DictReader(f)
        for l in c:
            if iseval:
                label = int(l['label'])
                labels.append(label)
            try:
                qtext = l['qtext'].decode('utf8')
                s_p_text = l['atext1'].decode('utf8')
                s_n_text = l['atext2'].decode('utf8')
            except AttributeError:  # python3 has no .decode()
                qtext = l['qtext']
                s_p_text = l['atext1']
                s_n_text = l['atext2']
            a_p_text = l['a1'].decode('utf8')
            a_n_text = l['a2'].decode('utf8')
            a_p.append(a_p_text.split(' '))
            a_n.append(a_n_text.split(' '))
            
            q.append(qtext.split(' '))
            s_p.append(s_p_text.split(' '))
            s_n.append(s_n_text.split(' '))
            q_sp.append(qtext.split(' ')+s_p_text.split(' '))
    if iseval:
        return (q, s_p, s_n, q_sp, a_p, a_n, np.array(labels))
    else:
        return (q, s_p, s_n, q_sp, a_p, a_n)
    
def make_model_inputs(qi, si_p, si_n, qi_si, ai_p, ai_n, f01, f10, f02, f20, f31, f13, f32, f23, 
                      q, s_p, s_n, q_sp, a_p, a_n, y=None):
    inp = {'qi': qi, 'si_p': si_p, 'si_n': si_n, 'qi_si':qi_si, 'ai_p':ai_p, 
          'ai_n':ai_n, 'f01':f01, 'f10':f10, 'f02':f02, 'f20':f20, 'f31':f31, 
          'f13':f13, 'f32':f32, 'f23':f23, 'q':q, 's_p':s_n, 's_n':s_n, 'q_sp':q_sp, 'a_p':a_p, 'a_n':a_n} 
    
    if y is not None:
        inp['y'] = y
    return inp

def load_set(fname, vocab=None, iseval=False):
    if iseval:
        q, s_p, s_n, q_sp, a_p, a_n, y = load_data_from_file(fname, iseval)
    else:
        q, s_p, s_n, q_sp, a_p, a_n = load_data_from_file(fname, iseval)
        vocab = utils.Vocabulary(q + s_p + s_n + a_p + a_n) 
    
    pad = conf['pad']
    
    qi = vocab.vectorize(q, pad=pad)  
    si_p = vocab.vectorize(s_p, pad=pad)
    si_n = vocab.vectorize(s_n, pad=pad)
    qi_si = vocab.vectorize(q_sp, pad=pad)
    ai_p = vocab.vectorize(a_p, pad=pad)
    ai_n = vocab.vectorize(a_n, pad=pad)
    
    f01, f10 = utils.sentence_flags(q, s_p, pad)  
    f02, f20 = utils.sentence_flags(q, s_n, pad)
    f31, f13 = utils.sentence_flags(q_sp, a_p, pad)
    f32, f23 = utils.sentence_flags(q_sp, a_n, pad)
    if iseval:
        inp = make_model_inputs(qi, si_p, si_n, qi_si, ai_p, ai_n, f01, f10, f02, f20, 
                                f31, f13, f32, f23, q, s_p, s_n, q_sp, a_p, a_n, y=y)
        return (inp, y)
    else:
        inp = make_model_inputs(qi, si_p, si_n, qi_si, ai_p, ai_n, f01, f10, f02, f20, 
                            f31, f13, f32, f23, q, s_p, s_n, q_sp, a_p, a_n)
        return (inp, vocab)        
    
def load_data(trainf, valf, testf):
    global vocab, inp_tr, inp_val, inp_test, y_val, y_test
    inp_tr, vocab = load_set(trainf, iseval=False)
    inp_val, y_val = load_set(valf, vocab=vocab, iseval=True)
    inp_test, y_test = load_set(testf, vocab=vocab, iseval=True)
    
def embedding():
    '''
    Declare all inputs (vectorized sentences and NLP flags)
    and generate outputs representing vector sequences with dropout applied.  
    Returns the vector dimensionality.       
    '''
    pad = conf['pad']
    dropout = conf['inp_e_dropout']
    
    # story selection
    input_qi = Input(name='qi', shape=(pad,), dtype='int32')                          
    input_si_p = Input(name='si_p', shape=(pad,), dtype='int32')                 
    input_f01 = Input(name='f01', shape=(pad, utils.flagsdim))
    input_f10 = Input(name='f10', shape=(pad, utils.flagsdim))

    input_si_n = Input(name='si_n', shape=(pad,), dtype='int32')  
    input_f02 = Input(name='f02', shape=(pad, utils.flagsdim))
    input_f20 = Input(name='f20', shape=(pad, utils.flagsdim))             

    # answer selection
    input_qi_si = Input(name='qi_si', shape=(pad,), dtype='int32')
    input_ai_p = Input(name='ai_p', shape=(pad,), dtype='int32')                        
    input_f31 = Input(name='f31', shape=(pad, utils.flagsdim))              
    input_f13 = Input(name='f13', shape=(pad, utils.flagsdim))          

    input_ai_n = Input(name='ai_n', shape=(pad,), dtype='int32')         
    input_f32 = Input(name='f32', shape=(pad, utils.flagsdim))            
    input_f23 = Input(name='f23', shape=(pad, utils.flagsdim))                       

    input_nodes = [input_qi, input_si_p, input_f01, input_f10, input_si_n,         
            input_f02, input_f20, input_qi_si, input_ai_p, input_f31, input_f13,
            input_ai_n, input_f32, input_f23]           
        
    N = emb.N + utils.flagsdim
    shared_embedding = Embedding(name='emb', input_dim=vocab.size(), input_length=pad,
                                output_dim=emb.N, mask_zero=False,
                                weights=[vocab.embmatrix(emb)], trainable=True)
    emb_qi_p = Dropout(dropout, noise_shape=(N,))(concatenate([shared_embedding(input_qi),
        input_f01]))
    emb_si_p = Dropout(dropout, noise_shape=(N,))(concatenate([shared_embedding(input_si_p),
        input_f10]))
    emb_qi_n = Dropout(dropout, noise_shape=(N,))(concatenate([shared_embedding(input_qi),
        input_f02]))
    emb_si_n = Dropout(dropout, noise_shape=(N,))(concatenate([shared_embedding(input_si_n),
        input_f20]))
    emb_qi_si_p = Dropout(dropout, noise_shape=(N,))(concatenate([shared_embedding(input_qi_si),
        input_f31]))
    emb_ai_p = Dropout(dropout, noise_shape=(N,))(concatenate([shared_embedding(input_ai_p),
        input_f13]))
    emb_qi_si_n = Dropout(dropout, noise_shape=(N,))(concatenate([shared_embedding(input_qi_si),
        input_f32]))
    emb_ai_n = Dropout(dropout, noise_shape=(N,))(concatenate([shared_embedding(input_ai_n),
        input_f23]))

    emb_outputs = [emb_qi_p, emb_si_p, emb_qi_n, emb_si_n, emb_qi_si_p, emb_ai_p, emb_qi_si_n, emb_ai_n]
    
    return N, input_nodes, emb_outputs
    
```
## Scoring Function

To handle the long sentences, the word level attention-based model is used as the scoring functions G and H.

The model builds the embeddings of two sequences of tokens X, Y. The model encodes each token of X, Y using a bidirectional LSTM and calculates the sentence vector X by applying a convolution on the output token vectors of the bidirectional LSTM on the X side. Then the each token vector of Y are multiplied by a softmax weight, which is determined by X. 

![fifth eq](https://latex.codecogs.com/gif.latex?%24%24m%28t%29%3Dtanh%28W_ah_y%28t%29&plus;W_qX%29%24%24) <br>
![sixth eq](https://latex.codecogs.com/gif.latex?%24%24o_t%20%5Cpropto%20exp%28w%5Et_%7Bms%7Dm%28t%29%29%24%24) <br>
![seventh eq](https://latex.codecogs.com/gif.latex?%24%24h%5E%5Cprime_y%28t%29%3Dh_y%28t%29o_t%24%24) <br>

where 
- ![eigth eq](https://latex.codecogs.com/gif.latex?%24h_y%28t%29%24) is the t-th token vector on the Y side.
- ![nineth eq](https://latex.codecogs.com/gif.latex?%24h%5E%5Cprime_y%28t%29%24) is the
updated t-th token vector. 
- ![tenth eq](https://latex.codecogs.com/gif.latex?%24W_a%2C%20W_q%2C%20w_%7Bms%7D%24) are attention
parameters

``` python
def attention_model(input_nodes, N, pfx=''):
    # apply biLSTM on each sentence X,Y
    qpos, pos, qneg, neg = rnn_input(N, pfx=pfx, dropout=conf['dropout'], dropoutfix_inp=conf['dropoutfix_inp'], 
                            dropoutfix_rec=conf['dropoutfix_rec'], sdim=conf['sdim'], 
                            rnnbidi_mode=conf['rnnbidi_mode'], rnn=conf['rnn'], rnnact=conf['rnnact'], 
                            rnninit=conf['rnninit'], inputs=input_nodes)
    
    # calculate the sentence vector on X side using Convolutional Neural Networks
    qpos_aggreg, qneg_aggreg, gwidth = aggregate(qpos, qneg, 'aggre_q'+pfx, N, 
                                               dropout=conf['dropout'], l2reg=conf['l2reg'], 
                                               sdim=conf['sdim'], cnnact=conf['cnnact'], cdim=conf['cdim'], 
                                               cfiltlen=conf['cfiltlen'])
    
    # re-embed X,Y in attention space
    awidth = int(N*conf['adim'])
    
    shared_dense_q = Dense(awidth, name='attn_proj_q'+pfx, kernel_regularizer=l2(conf['l2reg']))
    qpos_aggreg_attn = shared_dense_q(qpos_aggreg)
    qneg_aggreg_attn = shared_dense_q(qneg_aggreg)
    
    shared_dense_s = Dense(awidth, name='attn_proj_s'+pfx, kernel_regularizer=l2(conf['l2reg']))
    pos_attn = TimeDistributed(shared_dense_s)(pos)
    neg_attn = TimeDistributed(shared_dense_s)(neg)
    
    # apply an attention function on Y side by producing an vector of scalars denoting the attention for each token
    pos_foc, neg_foc = focus(N, qpos_aggreg_attn, qneg_aggreg_attn, pos_attn, neg_attn, 
                             pos, neg, conf['sdim'], awidth, 
                             conf['l2reg'], pfx=pfx)

    # calculate the sentence vector on Y side using Convolutional Neural Networks
    pos_aggreg, neg_aggreg, gwidth = aggregate(pos_foc, neg_foc, 'aggre_s'+pfx, N, 
                                  dropout=conf['dropout'], l2reg=conf['l2reg'], sdim=conf['sdim'],
                                  cnnact=conf['cnnact'], cdim=conf['cdim'], cfiltlen=conf['cfiltlen'])

    return ([qpos_aggreg, pos_aggreg], [qneg_aggreg, neg_aggreg]) 
    
def rnn_input(N, dropout=3/4, dropoutfix_inp=0, dropoutfix_rec=0,           
              sdim=2, rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', rnnbidi_mode=add, 
              inputs=None, pfx=''):
    if rnnbidi_mode == 'concat':
        sdim /= 2
    shared_rnn_f = rnn(int(N*sdim), kernel_initializer=rnninit, input_shape=(None, conf['pad'], N), 
                       activation=rnnact, return_sequences=True, dropout=dropoutfix_inp,
                       recurrent_dropout=dropoutfix_rec, name='rnnf'+pfx)
    shared_rnn_b = rnn(int(N*sdim), kernel_initializer=rnninit, input_shape=(None, conf['pad'], N),
                       activation=rnnact, return_sequences=True, dropout=dropoutfix_inp,
                       recurrent_dropout=dropoutfix_rec, go_backwards=True, name='rnnb'+pfx)
    qpos_f = shared_rnn_f(inputs[0])
    pos_f = shared_rnn_f(inputs[1])
    qneg_f = shared_rnn_f(inputs[2])
    neg_f = shared_rnn_f(inputs[3])
    
    qpos_b = shared_rnn_b(inputs[0])
    pos_b = shared_rnn_b(inputs[1])
    qneg_b = shared_rnn_b(inputs[2])
    neg_b = shared_rnn_b(inputs[3])

    qpos = Dropout(dropout, noise_shape=(conf['pad'], int(N*sdim)))(rnnbidi_mode([qpos_f, qpos_b]))
    pos = Dropout(dropout, noise_shape=(conf['pad'], int(N*sdim)))(rnnbidi_mode([pos_f, pos_b]))
    qneg = Dropout(dropout, noise_shape=(conf['pad'], int(N*sdim)))(rnnbidi_mode([qneg_f, qneg_b]))
    neg = Dropout(dropout, noise_shape=(conf['pad'], int(N*sdim)))(rnnbidi_mode([neg_f, neg_b]))
    
    return (qpos, pos, qneg, neg)

def aggregate(in1, in2, pfx, N, dropout, l2reg, sdim, cnnact, cdim, cfiltlen):
    '''
    In the paper, the sentence vector was calculated using simple averagring, 
    but we will use Convolutional Neural Networks in the demo.
    '''
    
    shared_conv = Convolution1D(name=pfx+'c', input_shape=(conf['pad'], int(N*sdim)), kernel_size=cfiltlen, 
                                filters=int(N*cdim), activation=cnnact, kernel_regularizer=l2(l2reg))
    aggreg1 = shared_conv(in1)
    aggreg2 = shared_conv(in2)

    nsteps = conf['pad'] - cfiltlen + 1
    width = int(N*cdim)
    
    aggreg1, aggreg2 = pool(pfx, aggreg1, aggreg2, nsteps, width, dropout=dropout)
    
    return (aggreg1, aggreg2, width)

def pool(pfx, in1, in2, nsteps, width, dropout):
    pooling = MaxPooling1D(pool_size=nsteps, name=pfx+'pool[0]')
    out1 = pooling(in1)
    out2 = pooling(in2)
    
    flatten = Flatten(name=pfx+'pool[1]')
    out1 = Dropout(dropout, noise_shape=(1, width))(flatten(out1))
    out2 = Dropout(dropout, noise_shape=(1, width))(flatten(out2))
    
    return (out1, out2)
    
def focus(N, input_aggreg1, input_aggreg2, input_seq1, input_seq2, orig_seq1, orig_seq2,
          sdim, awidth, l2reg, pfx=''):
    
    repeat_vec = RepeatVector(conf['pad'], name='input_aggreg1_rep'+pfx)
    input_aggreg1_rep = repeat_vec(input_aggreg1)
    input_aggreg2_rep = repeat_vec(input_aggreg2)
    
    attn1 = Activation('tanh')(add([input_aggreg1_rep, input_seq1]))
    attn2 = Activation('tanh')(add([input_aggreg2_rep, input_seq2]))
    
    shared_dense = Dense(1, kernel_regularizer=l2(l2reg), name='focus1'+pfx)
    attn1 = TimeDistributed(shared_dense)(attn1)
    attn2 = TimeDistributed(shared_dense)(attn2)
    
    flatten = Flatten(name='attn_flatten'+pfx)
    attn1 = flatten(attn1)
    attn2 = flatten(attn2)
    
    attn1 = Activation('softmax')(attn1)
    attn1 = RepeatVector(int(N*sdim))(attn1)
    attn1 = Permute((2,1))(attn1)
    output1 = multiply([orig_seq1, attn1])
    
    attn2 = Activation('softmax')(attn2)
    attn2 = RepeatVector(int(N*sdim))(attn2)
    attn2 = Permute((2,1))(attn2)
    output2 = multiply([orig_seq2, attn2])
    
    return (output1, output2)
```

To compare two sentence vectors, we used cosines similarity measure in the paper, but in the demo we use the mlp similarity function.

``` python
def mlp_ptscorer(inputs1, inputs2,  Ddim, N, l2reg, pfx='out', oact='sigmoid', extra_inp=[]):
    """ Element-wise features from the pair fed to an MLP. """

    sum1 = add(inputs1)
    sum2 = add(inputs2)
    mul1 = multiply(inputs1)
    mul2 = multiply(inputs2)

    mlp_input1 = concatenate([sum1, mul1])
    mlp_input2 = concatenate([sum2, mul2])

    # Ddim may be either 0 (no hidden layer), scalar (single hidden layer) or
    # list (multiple hidden layers)
    if Ddim == 0:
        Ddim = []
    elif not isinstance(Ddim, list):
        Ddim = [Ddim]
    if Ddim:
        for i, D in enumerate(Ddim):
            shared_dense = Dense(int(N*D), kernel_regularizer=l2(l2reg), 
                                 activation='tanh', name=pfx+'hdn[%d]'%(i,))
            mlp_input1 = shared_dense(mlp_input1)
            mlp_input2 = shared_dense(mlp_input2)

    shared_dense = Dense(1, kernel_regularizer=l2(l2reg), activation=oact, name=pfx+'mlp')
    mlp_out1 = shared_dense(mlp_input1)
    mlp_out2 = shared_dense(mlp_input2)
    
    return [mlp_out1, mlp_out2]    
```

## Model Architecture

``` python
def build_model():
    # input embedding         
    N, input_nodes_emb, output_nodes_emb = embedding()
    
    # story selection
    ptscorer_inputs1, ptscorer_inputs2 = avg_model(output_nodes_emb[:4], N, pfx='S')

    scoreS1, scoreS2 = mlp_ptscorer(ptscorer_inputs1, ptscorer_inputs2, conf['Ddim'], N,  
            conf['l2reg'], pfx='outS', oact='sigmoid')                

    # anwer selection
    ptscorer_inputs3, ptscorer_inputs4 = avg_model(output_nodes_emb[4:], N, pfx='A')
    
    scoreA1, scoreA2 = mlp_ptscorer(ptscorer_inputs3, ptscorer_inputs4, conf['Ddim'], N,
            conf['l2reg'], pfx='outA', oact='sigmoid')

    output_nodes = [scoreS1, scoreS2, scoreA1, scoreA2]

    model = Model(inputs=input_nodes_emb, outputs=output_nodes)
    
    model.compile(loss=ranking_loss, optimizer=conf['opt'])
    return model
    
``` 

## Loss Function
Training is performed with a hinge rank loss over these two triplets:

![eleventh eq](https://latex.codecogs.com/gif.latex?%24%24%5Csum_%7Bs_i%20%5Cneq%20s%5E*%7D%5E%7B%7CX%7C%7D%20max%280%2C%20%5Cgamma_s%20-%20G%28q%2Cs%5E*%29%20&plus;%20G%28q%2Cs_i%29%29%20&plus;%20%5Csum_%7Ba_r%20%5Cneq%20a%5E*%7D%5E%7Bk%7D%20max%280%2C%20%5Cgamma_a%20-%20H%28s_a%2Cs%5E*%29%20&plus;%20H%28s_a%2C%20a_r%29%29%24%24)

where 
- ![twelveth eq](https://latex.codecogs.com/gif.latex?%24s%5E*%24) is the correct relevant story for q, i.e. ![thirteenth eq](https://latex.codecogs.com/gif.latex?%24s%5E*%20%3D%20e_c%20%7C%7C%20l_c%24)
- ![fourteenth eq](https://latex.codecogs.com/gif.latex?%24a%5E*%24) is the correct answer sentence for q. 
- ![fifteenth eq](https://latex.codecogs.com/gif.latex?y_s) and ![sixteenth eq](https://latex.codecogs.com/gif.latex?y_a) are margins 

``` python
'''
posS: G(q, s^*)
negS: G(q, s_i)
posA: H(s_a, s^*)
negA: H(s_a, a_r)
'''
def ranking_loss(y_true, y_pred):
    posS = y_pred[0]
    negS = y_pred[1]
    posA = y_pred[2]
    negA = y_pred[3]

    margin = conf['margin']
    loss = K.maximum(margin + negS - posS, 0.0) + K.maximum(margin + negA - posA, 0.0) 
    return K.mean(loss, axis=-1)
    
```

## Train and Evaluation

``` python

def train_and_eval(runid):
    print('Model')
    model = build_model()
    print(model.summary())
    
    print('Training')
    fit_model(model, weightsf='weights-'+runid+'-bestval.h5')
    model.save_weights('weights-'+runid+'-final.h5', overwrite=True)
    model.load_weights('weights-'+runid+'-bestval.h5')

    print('Predict&Eval (best val epoch)')
    res = eval(model)
    
def fit_model(model, **kwargs):
    epochs = conf['epochs']
    callbacks = fit_callbacks(kwargs.pop('weightsf'))
    
    # During the computation, these values will not be used at all.
    # Note that the variable 'y_true' in function ranking_loss does not participate in calculations.
    dummy1 = np.ones((len(inp_tr['qi']),1), dtype=np.float) 
    dummy2 = np.ones((len(inp_val['qi']),1), dtype=np.float)
    
    return model.fit(inp_tr, y=[dummy1,dummy1,dummy1,dummy1], validation_data=[inp_val,
        [dummy2,dummy2,dummy2,dummy2]], callbacks = callbacks, epochs=epochs)
```

At every epoch, the callback function measures mrr performance and accuracy 

``` python
def fit_callbacks(weightsf):                                  
    return [utils.AnsSelCB(inp_val['q'], inp_val['s_p'], inp_val['s_n'], inp_val['q_sp'], 
        inp_val['a_p'], inp_val['a_n'], y_val, inp_val),
            ModelCheckpoint(weightsf, save_best_only=True, monitor='acc', mode='max'),
            EarlyStopping(monitor='acc', mode='max', patience=12)]
            
def eval(model):
    res = []
    for inp in [inp_val, inp_test]:
        if inp is None:
            res.append(None)
            continue

        pred = model.predict(inp)
        ypredS = pred[0]
        ypredA1 = pred[2]
        ypredA2 = pred[3]

        res.append(utils.eval_QA(ypredS, ypredA1, ypredA2, inp['q'], inp['y'], MAP=False))
    return tuple(res)
    
if __name__ == "__main__":
    trainf = 'data/anssel/pororo/train_triplet_concat_a5_500.csv' 
    valf = 'data/anssel/pororo/dev_triplet_concat_a5_for_mrr_500.csv'
    testf = 'data/anssel/pororo/dev_triplet_concat_a5_for_mrr_500.csv'
    params = []
    
    conf, ps, h = config()

    if conf['emb'] == 'Glove':
        print('GloVe')
        emb = utils.GloVe(N=conf['embdim'])

    print('Dataset')
    load_data(trainf,valf,testf)
    runid = 'DEMN-%x' % (h)
    print('RunID: %s  (%s)' % (runid, ps))
    train_and_eval(runid)
