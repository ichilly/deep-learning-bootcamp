import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob) # add dropout

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob) # add dropout

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
hypothesis = tf.nn.dropout(L3, keep_prob=keep_prob) # add dropout

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                              logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}  # add keep_prob
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

#Epoch: 0001 cost = 206.500136501
#Epoch: 0002 cost = 4.046328312
#Epoch: 0003 cost = 2.883343892
#Epoch: 0004 cost = 2.592236231
#Epoch: 0005 cost = 2.471184444
#Epoch: 0006 cost = 2.431030469
#Epoch: 0007 cost = 2.348688173
#Epoch: 0008 cost = 2.354123566
#Epoch: 0009 cost = 2.334435292
#Epoch: 0010 cost = 2.344501695
#Epoch: 0011 cost = 2.316332093
#Epoch: 0012 cost = 2.340394587
#Epoch: 0013 cost = 2.302586533
#Epoch: 0014 cost = 2.304660596
#Epoch: 0015 cost = 2.312667777
#Learning Finished!
