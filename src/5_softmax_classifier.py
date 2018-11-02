import tensorflow as tf

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

nb_classes = 3  # number of classes need to be classified
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # print the classification results
    all_h = sess.run(hypothesis, feed_dict={X: [[1,2,1,1],[4,1,5,5], [1,6,6,6]]})
    print(all_h, sess.run(tf.arg_max(all_h, 1)))
    
    # predict the result for X: [[1, 11, 7, 9]]
    h = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(h, sess.run(tf.arg_max(h, 1)))

# Output (step, cost)
#0 4.765922
#200 0.6290847
#400 0.51537645
#600 0.4170147
#800 0.32317644
#1000 0.251263
#1200 0.22613744
#1400 0.20566568
#1600 0.18849207
#1800 0.17387868
#2000 0.16129808
#
# X=[1,2,1,1], Y=[0,0,1], Y is classified to 2
# X=[4,1,5,5], Y=[0,1,0], Y is classified to 1
# X=[1,2,1,1], Y=[1,0,0], Y is classified to 0
#[[1.7265975e-06 7.1256905e-04 9.9928576e-01]
# [6.5697081e-06 8.4183824e-01 1.5815522e-01]
# [7.4919897e-01 2.5079495e-01 6.0673619e-06]] [2 1 0]
#
# Predict result for X: [[1, 11, 7, 9]]
#[[9.0370718e-03 9.9095726e-01 5.6999979e-06]] [1]
