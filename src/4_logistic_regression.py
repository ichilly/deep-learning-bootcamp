import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1, 1 + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

# use GradientDescentOptimizer to train
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# greater than 0.5 means true, else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# train the model
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nPredicted (Y): ", c, "\nAccuracy: ", a)

# Output (step, cost)
#0 0.71963215
#200 0.4582506
#400 0.35679486
#600 0.312078
#800 0.28838024
#1000 0.2735134
#1200 0.2628253
#1400 0.25432715
#1600 0.24708746
#1800 0.24063753
#2000 0.23472738
#2200 0.22921604
#2400 0.22402012
#2600 0.21908762
#2800 0.21438402
#3000 0.2098853
#3200 0.2055735
#3400 0.20143463
#3600 0.19745709
#3800 0.19363095
#4000 0.18994755
#4200 0.18639904
#4400 0.18297833
#4600 0.17967878
#4800 0.1764944
#5000 0.17341937
#5200 0.17044844
#5400 0.16757673
#5600 0.16479938
#5800 0.16211207
#6000 0.15951067
#6200 0.1569913
#6400 0.15455012
#6600 0.15218385
#6800 0.14988908
#7000 0.14766277
#7200 0.14550185
#7400 0.14340375
#7600 0.14136575
#7800 0.13938536
#8000 0.13746017
#8200 0.13558803
#8400 0.13376679
#8600 0.13199444
#8800 0.13026902
#9000 0.12858875
#9200 0.12695193
#9400 0.12535693
#9600 0.123802125
#9800 0.12228608
#10000 0.12080735
#
#Hypothesis:  [[0.03313247]
#              [0.16196638]
#              [0.31611842]
#              [0.776289  ]
#              [0.9363201 ]
#              [0.9790829 ]]
#Predicted (Y):  [[0.]
#                [0.]
#                [0.]
#                [1.]
#                [1.]
#                [1.]]
#Accuracy:  1.0
