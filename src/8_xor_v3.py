import numpy as np
import tensorflow as tf

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

width = 10

W1 = tf.Variable(tf.random_normal([2,width]), name='weight1')
b1 = tf.Variable(tf.random_normal([width]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([width,width]), name='weight2')
b2 = tf.Variable(tf.random_normal([width]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# ===Add one more layer here===
W3 = tf.Variable(tf.random_normal([width,width]), name='weight3')
b3 = tf.Variable(tf.random_normal([width]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([width,1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2, W3, W4])) # run all layers

    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nPredicted(Y): ", c, "\nAccuracy: ", a)

#Hypothesis:  [[0.23067099]
#              [0.6648481 ]
#              [0.6621891 ]
#              [0.47235376]]
#Predicted(Y):  [[0.]
#                [1.]
#                [1.]
#                [0.]] 
#Accuracy:  1.0
