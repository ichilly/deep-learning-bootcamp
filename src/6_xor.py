import numpy as np
import tensorflow as tf

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nPredicted(Y): ", c, "\nAccuracy: ", a)

# Output(step, cost, W)
#0 1.0027125 [[-0.2713297]
#             [ 1.3324895]]
#1000 0.70626515 [[-0.47068843]
#                 [ 0.4408861 ]]
#2000 0.6969285 [[-0.26481688]
#                [ 0.22604445]]
#3000 0.69424033 [[-0.14486393]
#                 [ 0.11829468]]
#4000 0.6934627 [[-0.07943001]
#                [ 0.06147379]]
#5000 0.6932385 [[-0.04377292]
#                [ 0.03164387]]
#6000 0.6931738 [[-0.02427705]
#                [ 0.01608453]]
#7000 0.69315505 [[-0.01356678]
#                 [ 0.00803326]]
#8000 0.6931495 [[-0.0076485 ]
#                [ 0.00391096]]
#9000 0.6931479 [[-0.0043553 ]
#                [ 0.00183084]]
#10000 0.6931474 [[-0.00250782]
#                 [ 0.00080273]]
#
#Hypothesis:  [[0.5002528 ]
#              [0.5004535 ]
#              [0.49962583]
#              [0.49982655]]
#Predicted(Y):  [[1.]
#                [1.]
#                [0.]
#                [0.]] 
#Accuracy:  0.5
