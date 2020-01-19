# Week-1

Welcome to this fifth course on deep learning. In this course, you learn about sequence models, one of the most exciting areas in deep learning. Models like recurrent neural networks or RNNs have transformed speech recognition, natural language processing and other areas. And in this course, you learn how to build these models for yourself. 

Let's start by looking at a few examples of where sequence models can be useful. `In speech recognition` you are given an `input audio clip X` and asked to map it to a `text transcript Y`. Both the input and the output here are sequence data, because X is an audio clip and so that plays out over time and Y, the output, is a sequence of words. So sequence models such as a recurrent neural networks and other variations, you'll learn about in a little bit have been very useful for speech recognition. 

`Music generation` is another example of a problem with sequence data. In this case, only the `output Y is a sequence`, the input can be the empty set, or it can be a single integer, maybe referring to the genre of music you want to generate or maybe the first few notes of the piece of music you want. But here X can be nothing or maybe just an integer and output Y is a sequence. 

`In sentiment classification` the `input X is a sequence`, so given the input phrase like, "There is nothing to like in this movie" how many stars do you think this review will be? 

Sequence models are also very useful for `DNA sequence analysis`. So your DNA is represented via the four alphabets A, C, G, and T. And so `given a DNA sequence` can you label which part of this DNA sequence say corresponds to a protein. 

`In machine translation` you are given an input sentence, voulez-vou chante avec moi? And you're asked to output the translation in a different language. 

`In video activity recognition` you might be given `a sequence of video frames` and asked to recognize the activity. 

And `in name entity recognition` you might be given a sentence and asked to identify the people in that sentence. So all of these problems can be addressed as supervised learning with label data X, Y as the training set. 

But, as you can tell from this list of examples, there are a lot of different types of sequence problems. In some, both the input X and the output Y are sequences, and in that case, sometimes X and Y can have different lengths, or in this example and this example, X and Y have the same length. And in some of these examples only either X or only the opposite Y is a sequence. So in this course you learn about sequence models are applicable, so all of these different settings. 


## Different types of RNNs architectures

- one to one 
- one to many: Music Generation 
- many to one: Sentiment Classification
- many to many (with the lengths of input and output sequences being the same or different ): Machine Translation, DNS Sequence Analysis

![](https://github.com/xuyanMax/image-cache/blob/master/rnn/rnn_architect_summary.png)
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/rnn_architec1.png)
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/rnn_architec2.png)

## Language model and sequence generation
Language model predicts the possibilities of the next sentences of a particular sentence.

## Language modeling using RNN
Training set: a large corpus of English text
Tokenize the English sentence, which means to form a <vocabulary, index> mapping using one-hot vectors.

When sentences end, it is common to add and extra token called `EOS` (End of Sentence)

### Build an RNN to model the different chances of sequences
So each step in the RNN will look at some set of preceding words such as, given the first three words, what is the distribution over the next word? And so this RNN learns to predict one word at a time going from left to right.

![](https://github.com/xuyanMax/image-cache/blob/master/rnn/rnn_build_model.png)

At a certain time, t, if the true word was yt and the new networks soft max predicted some y hat t, then this is the `soft max loss function` that you should already be familiar with. And then the `overall loss` is just the sum 
overall time steps of the loss associated with the individual predictions. 

The probability P of given a new sentence (y1, y2, y3): 
- p1: the first soft max gives your the chance of y1
- p2: the second one tells you the chance of y2 given y1
- p3: the third one gives you the chance of y3 given (y1, y2)
- P = p1 * p2 * p3

### Sampling novel sequences
#### Sample sequences from the RNN Model.

After you train a sequence model, one of the ways you can informally get a sense of what is learned is to have a sample novel sequences.

So the network was trained using this structure shown at the top. But to sample, you do something slightly different, so what you want to do is `first sample what is the first word you want your model to generate`. And so for that you input the `usual x1 equals 0`, `a0 equals 0`. And now your first time stamp will have some max probability over possible outputs. So what you do is you then `randomly sample` according to this soft max distribution. So what the soft max distribution gives you is it tells you what is the chance that it refers to this a, what is the chance that it refers to this Aaron? What's the chance it refers to Zulu, what is the chance that the first word is the Unknown word token. Maybe it was a chance it was a end of sentence token. And then you take this vector and use, for example, the numpy command `np.random.choice` to sample according to distribution defined by this vector probabilities, and that lets you sample the first words. Next you then go on to the second time step, and now remember that the second time step is expecting this `y1 as input`. But what you do is you then take the y1 hat that you just sampled and pass that in here as the input to the next timestamps. So whatever works, you just chose the first time step passes this input in the second position, and then this soft max will make a prediction for what is y hat 2. 
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/rnn_sampling.png)

#### Word-Level RNN vs. Character-Level RNN
For word level language model, your vocabulary contains words like, `[a, aarun, ...zulu, <UNK>]`. Whereas for character language model, your vocabulary contains characters like '[a,b,c,...z,A,B,C,...Z,0,1,2,...9,.,;, ,‘,...]'.

Using character-level RNN has some pros and cons.

The advantages of character-level RNN:
- No need to worry about unknown word tokens. It is able to assign a sequence like `mau` non-zero probability. Whereas if `mau` is out of your vocabulary for the word-level language model, you just have to assign it the unknown word token.

The `drawbacks` of using character-level RNN includes:
- You end up with much longer sequences (Many English sentence will have 10-2- words but may have many dozens of of characters, like (a,b,c,...,z, A,B,C,...,Z,0,1,2,...,9,;, ,/,...))
- It is more hardware, computationally expensive to train, which is not widely used today.
- It is not as good as word-level language model at `capturing long range dependencies` between how the earlier parts of the sentence also affect the later parts of the sentence.
- Used in specialized applications where you might need to deal with unknown words a lot or where you have a specialized vocabulary.
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/rnn_character_level_language_model.png)
