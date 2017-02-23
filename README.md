## Convolutional Neural Networks for Text Classification for binary-tags augmented sentences
This is a fork from [Yoon Kim repository](https://github.com/yoonkim/CNN_sentence) for sentence classification using Convolutional Neural Networks.

The input to the network are sentences where each token has been labeled with a binary tag (for instance a Named Entity Recognition tag).
The label of each token is then appended to its word vectors and used during the training of the CNN.

### Requirements
Code is written in Python (2.7) and requires Theano (0.7).

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/


### Data Preprocessing
The raw data is assumed to be the (tab-seprated) juxaposition of the sentence and its associated label sequence. See the files `dataset.pos` and `dataset.neg` for an example.

To process the raw data, run

```
python process_data.py
```

where path points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file). 
This will create a pickle object called `mr.p` in the same folder.


### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -rand
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -static -word2vec
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

This will run the CNN-rand, CNN-static, and CNN-nonstatic models respectively in the paper.

### Using the GPU
GPU will result in a good 10x to 20x speed-up, so it is highly recommended. 
To use the GPU, simply change `device=cpu` to `device=gpu` (or whichever gpu you are using).
For example:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

