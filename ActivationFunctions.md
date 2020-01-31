# Activations

### Sigmoid 
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/sigmoid.png)
### Tanh
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/tanh.png)
### ReLU
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/relu.png)
### Swish
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/swish.png)
### Softmax
![](https://github.com/xuyanMax/image-cache/blob/master/rnn/softmax.png)

## Choosing the right Activation Function
- Sigmoid functions and their combinations generally work better in the case of classifiers
- Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
- ReLU function is a general activation function and is used in most cases these days
- If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice
- Always keep in mind that ReLU function should only be used in the hidden layers
- As a rule of thumb, you can begin with using ReLU function and then move over to other activation functions in case ReLU doesn’t provide with optimum results
