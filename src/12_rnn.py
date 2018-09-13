import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # reproducibility

# Teach hello: hihell -> ihello

# Data Declaration
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]    # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello
num_classes = 5
input_dim = 5        # one-hot size
hidden_size = 5      # output from the LSTM. 5 to directly predict one-hot
batch_size = 1       # one sentence
sequence_length = 6  # |ihello| == 6
learning_rate = 0.1

# Feed to RNN
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])    # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])                 # Y label
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# Sequence loss
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "Y: ", y_data)
        
        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))

#0 loss: 1.6107724 prediction:  [[2 3 3 4 4 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ellooo
#1 loss: 1.5258473 prediction:  [[2 3 3 3 4 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ellloo
#2 loss: 1.4532484 prediction:  [[2 3 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elello
#3 loss: 1.3810189 prediction:  [[2 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elelll
#4 loss: 1.3161988 prediction:  [[2 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elelll
#5 loss: 1.2625375 prediction:  [[2 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elelll
#6 loss: 1.2124075 prediction:  [[2 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elelll
#7 loss: 1.1680179 prediction:  [[2 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elelll
#8 loss: 1.1323599 prediction:  [[2 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elelll
#9 loss: 1.1045315 prediction:  [[2 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elelll
#10 loss: 1.0788342 prediction:  [[2 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  elelll
#11 loss: 1.0517924 prediction:  [[3 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  llelll
#12 loss: 1.0240762 prediction:  [[3 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  llelll
#13 loss: 0.9967444 prediction:  [[3 3 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  llelll
#14 loss: 0.9694516 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#15 loss: 0.9433968 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#16 loss: 0.9203515 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#17 loss: 0.9002113 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#18 loss: 0.8839547 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#19 loss: 0.8709547 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#20 loss: 0.86041605 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#21 loss: 0.85245246 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#22 loss: 0.84556746 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#23 loss: 0.84029555 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#24 loss: 0.8354871 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#25 loss: 0.8314445 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#26 loss: 0.82763547 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#27 loss: 0.82402956 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#28 loss: 0.82054013 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#29 loss: 0.816898 prediction:  [[3 0 2 3 3 3]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  lhelll
#30 loss: 0.8133373 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#31 loss: 0.80947375 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#32 loss: 0.8056714 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#33 loss: 0.8015649 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#34 loss: 0.79752487 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#35 loss: 0.7931895 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#36 loss: 0.7888834 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#37 loss: 0.78424305 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#38 loss: 0.7794771 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#39 loss: 0.77419245 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#40 loss: 0.7685175 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#41 loss: 0.7622608 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#42 loss: 0.7557626 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#43 loss: 0.7491845 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#44 loss: 0.7429266 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#45 loss: 0.73718953 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#46 loss: 0.7319406 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#47 loss: 0.727174 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#48 loss: 0.722571 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
#49 loss: 0.7180599 prediction:  [[1 0 2 3 3 4]] Y:  [[1, 0, 2, 3, 3, 4]]
#    Prediction str:  ihello
