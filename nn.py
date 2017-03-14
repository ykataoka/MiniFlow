"""
Check out the new network architecture and dataset!

Notice that the weights and biases are
generated randomly.

No need to change anything, but feel free to tweak
to test your network, play around with the epochs, batch size, etc!
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniFlow import *

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 1000
# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables, 0.0005)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))


#     
# 
# import numpy as np
# from miniflow import *
# 
# X, W, b = Input(), Input(), Input()
# y = Input()
# f = Linear(X, W, b)
# a = Sigmoid(f)
# cost = MSE(y, a)
# 
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2.], [3.]])
# b_ = np.array([-3.])
# y_ = np.array([1, 2])
# 
# feed_dict = {
#     X: X_,
#     y: y_,
#     W: W_,
#     b: b_,
# }
# 
# graph = topological_sort(feed_dict)
# forward_and_backward(graph)
# # return the gradients for each Input
# gradients = [t.gradients[t] for t in [X, y, W, b]]

"""
Expected output

[array([[ -3.34017280e-05,  -5.01025919e-05],
       [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
       [ 1.9999833]]), array([[  5.01028709e-05],
       [  1.00205742e-04]]), array([ -5.01028709e-05])]
"""


"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""


"""
Task 1
"""
#
#from miniFlow import *
#
#x, y = Input(), Input()
#
#f = Add(x, y)
#
#feed_dict = {x: 10, y: 5}
#
#sorted_nodes = topological_sort(feed_dict)
#output = forward_pass(f, sorted_nodes)
#
## NOTE: because topological_sort set the values for the `Input` nodes we could also access
## the value for x with x.value (same goes for y).
#print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))



"""
NOTE: Here we're using an Input node for more than a scalar.
In the case of weights and inputs the value of the Input node is
actually a python list!
In general, there's no restriction on the values that can be
passed to an Input node.
"""
# from miniFlow import *
# 
# inputs, weights, bias = Input(), Input(), Input()
# 
# f = Linear(inputs, weights, bias)
# 
# feed_dict = {
#     inputs: [6, 14, 3],
#     weights: [0.5, 0.25, 1.4],
#     bias: 2
# }
# 
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
# 
# print(output) # should be 12.7 with this example


"""
The setup is similar to the prevous `Linear` node you wrote
except you're now using NumPy arrays instead of python lists.

Update the Linear class in miniflow.py to work with
numpy vectors (arrays) and matrices.

Test your code here!
"""

# import numpy as np
# from miniFlow import *
# 
# X, W, b = Input(), Input(), Input()
# 
# f = Linear(X, W, b)
# 
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2., -3], [2., -3]])
# b_ = np.array([-3., -5])
# 
# X_ = np.array([[8., 0.], [4, 4]])
# W_ = np.array([[7., -1], [4., -10]])
# b_ = np.array([2., -1])
# 
# #inputs: [[8 0] [4 4]], weights: [[ 7 -1] [ 4 -10]], bias: [ 2 -1]
# 
# feed_dict = {X: X_, W: W_, b: b_}
# 
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
# 
# """
# Output should be:
# [[-9., 4.],
# [-9., 4.]]
# """
# print(output)


"""
This network feeds the output of a linear transform
to the sigmoid function.

Finish implementing the Sigmoid class in miniflow.py!

Feel free to play around with this network, too!
"""
# import numpy as np
# from miniFlow import *
# 
# X, W, b = Input(), Input(), Input()
# 
# f = Linear(X, W, b)
# g = Sigmoid(f)
# 
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2., -3], [2., -3]])
# b_ = np.array([-3., -5])
# 
# feed_dict = {X: X_, W: W_, b: b_}
# 
# graph = topological_sort(feed_dict)
# output = forward_pass(g, graph)
# 
# """
# Output should be:
# [[  1.23394576e-04   9.82013790e-01]
#  [  1.23394576e-04   9.82013790e-01]]
# """
# print(output)
