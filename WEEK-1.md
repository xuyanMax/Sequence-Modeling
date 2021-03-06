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

When sentences end, it is common to add an extra token called `EOS` (End of Sentence)

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

So the network was trained using this structure shown at the top. But to sample, you do something slightly different, so what you want to do is first sample what is the first word you want your model to generate. And so for that you input the `usual x1 equals 0`, `a0 equals 0`. And now your first time stamp will have some max probability over possible outputs. So what you do is you then `randomly sample` according to this soft max distribution. So what the soft max distribution gives you is it tells you what is the chance that it refers to this a, what is the chance that it refers to this Aaron? What's the chance it refers to Zulu, what is the chance that the first word is the Unknown word token. Maybe it was a chance of a token of the end of sentence. 

And then you take this vector and use, for example, the numpy command `np.random.choice` to sample according to distribution defined by this vector probabilities, and that lets you sample the first words. Next you then go on to the second time step, and now remember that the second time step is expecting this `y1 as input`. But what you do is you then take the y1 hat that you just sampled and pass that in here as the input to the next timestamps. So whatever works, you just chose the first time step passes this input in the second position, and then this soft max will make a prediction for what is y hat 2. 
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/rnn_sampling.png)

#### Word-Level RNN vs. Character-Level RNN
For word level language model, your `vocabulary` contains words like, `[a, aarun, ...zulu, <UNK>]`. Whereas for character language model, your vocabulary contains characters like '[a,b,c,...z,A,B,C,...Z,0,1,2,...9,.,;, ,‘,...]'.

**Using character-level RNN has some pros and cons.**

The advantages of character-level RNN:
- No need to worry about unknown word tokens. It is able to assign a sequence like `mau` non-zero probability. Whereas if `mau` is out of your vocabulary for the word-level language model, you just have to assign it the unknown word token.

The drawbacks of using character-level RNN includes:
- You end up with much longer sequences (Many English sentence will have 10-20 words but may have many dozens of characters, like (a,b,c,...,z, A,B,C,...,Z,0,1,2,...,9,;, ,/,...))
- It is more hardware, computationally expensive to train, which is not widely used today.
- It is not as good as word-level language model at `capturing long range dependencies` between how the earlier parts of the sentence also affect the later parts of the sentence.
- Used in specialized applications where you might need to deal with unknown words a lot or where you have a specialized vocabulary.
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/rnn_character_level_language_model.png)

### Vanishing gradients with RNN
#### What is exploding gradients
When activation functions are used whose derivatives can take on larger values, one risks encountering the related exploding gradient problem.

#### What is vanishing gradients
In machine learning, the vanishing gradient problem is a difficulty found in training artificial neural networks with gradient-based learning methods and back-propagation. In such methods, each of the neural network's weights receives an update proportional to the partial derivative of the error function with respect to the current weight in each iteration of training. The problem is that in some cases, the gradient will be vanishingly small, effectively preventing the weight from changing its value. In the worst case, this may completely stop the neural network from further training. As one example of the problem cause, traditional activation functions such as the hyperbolic tangent function have gradients in the range (-1, 1), and back-propagation computes gradients by the chain rule. This has the effect of multiplying n of these small numbers to compute gradients of the "front" layers in an n-layer network, meaning that the gradient (error signal) decreases exponentially with n while the front layers train very slowly.

#### Vanishing gradients with RNN
But it turns out the basics RNN we've seen so far it's `not` very good at capturing very long-term dependencies. To explain why, you might remember from our early discussions of training very deep neural networks, that we talked about the `vanishing gradients problem`. So there is a very, very deep neural network say, 100 layers or even much deeper than you would carry out forward prop, from left to right and then back prop. And we said that, if this is a very deep neural network, then the gradient from just output y, would have a very hard time propagating back to affect the weights of these earlier layers, to affect the computations in the earlier layers. 

And for an RNN with a similar problem, you have forward prop came from left to right, and then back prop, going from right to left. And it can be quite difficult, because of the same vanishing gradients problem, for the outputs of the errors associated with the later time steps to affect the computations that are earlier. And so in practice, what this means is, it might be difficult to get a neural network to realize that it needs to memorize a singular noun or a plural noun, so that later on in the sequence that can generate either was or were, depending on whether it was singular or plural. And notice that in English, this stuff in the middle could be arbitrarily long, right? So you might need to memorize the singular/plural for a very long time before you get to use that bit of information. 

So because of this problem, the basic RNN model has many `local influences`, meaning that the output y^<3> is mainly influenced by values close to y^<3>. And a value here is mainly influenced by inputs that are somewhere close. And it's difficult for the output here to be strongly influenced by an input that was very early in the sequence. And this is because whatever the output is, whether this got it right, this got it wrong, it's just very difficult for the area to back propagate all the way to the beginning of the sequence, and therefore to modify how the neural network is doing computations earlier in the sequence. 

What can we do about it?    
### Gated Recurrent Unit(GRU)
 Gated Recurrent Unit which is a modification to the RNN hidden layer that makes it much better at capturing long range connections and helping a lot with the vanishing gradient problems.

![](https://github.com/xuyanMax/image-cache/blob/master/rnn/GRU.png)

### Long Short Term Memory(LSTM)
According to [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by `Hochreiter & Schmidhuber` (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![](https://github.com/xuyanMax/image-cache/blob/master/rnn/lstm.png)

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called `gates`.

`Gates` are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

![](https://github.com/xuyanMax/image-cache/blob/master/rnn/pointwise.png)

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

**An LSTM has three of these gates(forget gate, update gate & output gate), to protect and control the cell state.**

#### Step-by-Step LSTM Walk Through
The first step in our LSTM is to `decide what information we’re going to throw away from the cell state`. This decision is made by a sigmoid layer called the “forget gate layer.”
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/lstm_step1.png)

The next step is to `decide what new information we’re going to store in the cell state`. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, [Math Processing Error], that could be added to the state. In the next step, we’ll combine these two to create an update to the state.
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/lstm_step2.png)

Thirdly, it’s now time to `update the old cell state`, [Math Processing Error], into the new cell state [Math Processing Error]. The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by [Math Processing Error], forgetting the things we decided to forget earlier. Then we add [Math Processing Error]. This is the new candidate values, scaled by how much we decided to update each state value.
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/lstm_step3.png)

Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through [Math Processing Error] and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/lstm_step4.png)


### Bidirectional RNN

![](https://github.com/xuyanMax/image-cache/blob/master/rnn/brnn.png)
For a lots of NLP problems, for a lot of text with natural language processing problems, a bidirectional RNN with a LSTM appears to be commonly used.

The disadvantage of the bidirectional RNN is that `you do need the entire sequence of data before you can make predictions anywhere`. So, for example, if you're building a speech recognition system, then the BRNN will let you take into account the entire speech utterance but if you use this straightforward implementation, you need to wait for the person to stop talking to get the entire utterance before you can actually process it and make a speech recognition prediction. So for a real type speech recognition applications, they're somewhat more complex modules as well rather than just using the standard bidirectional RNN as you've seen here. But for a lot of natural language processing applications where you can get the entire sentence all the same time, the standard BRNN algorithm is actually very effective. 

### Deep RNNs
The different versions of RNNs you've seen so far will already work quite well by themselves. But for learning very complex functions sometimes is useful to stack multiple layers of RNNs together to build even deeper versions of these models. 

But I've changed the notation a little bit which is that, instead of writing this as a0 for the activation time zero, I've added this square bracket 1 to denote that this is for layer one. So the notation we're going to use is a[l] to denote that it's an activation associated with layer l and then <t> to denote that that's associated over time t.

![](https://github.com/xuyanMax/image-cache/blob/master/rnn/deep_rnn.png)