import numpy as np
import tensorflow as tf

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# ===Add a layer here===
W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
# Original depth
W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2])) # run both layers


    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nPredicted(Y): ", c, "\nAccuracy: ", a)

# Output (step, cost, W1, W2)
#0 0.8422477 [array([[ 1.2392795 , -0.66471064],
#                    [ 0.36096406,  0.38395777]], dtype=float32), array([[ 0.8957559 ],
#                                                                        [-0.06977098]], dtype=float32)]
#1000 0.6922335 [array([[ 1.2389368 , -0.65737927],
#                       [ 0.38779065,  0.3793155 ]], dtype=float32), array([[ 0.36514091],
#                                                                           [-0.12578589]], dtype=float32)]
#2000 0.69197196 [array([[ 1.2454268 , -0.65318125],
#                        [ 0.41840675,  0.37117106]], dtype=float32), array([[ 0.3774543 ],
#                                                                            [-0.10193514]], dtype=float32)]
#3000 0.6917133 [array([[ 1.2527682 , -0.64994866],
#                       [ 0.45011267,  0.3647819 ]], dtype=float32), array([[ 0.39702153],
#                                                                           [-0.07817281]], dtype=float32)]
#4000 0.69143534 [array([[ 1.2610229 , -0.64758396],
#                        [ 0.4830404 ,  0.36010095]], dtype=float32), array([[ 0.4183514 ],
#                                                                            [-0.05515739]], dtype=float32)]
#5000 0.69113314 [array([[ 1.2702899 , -0.6460362 ],
#                        [ 0.5173012 ,  0.35704246]], dtype=float32), array([[ 0.44145253],
#                                                                            [-0.03274345]], dtype=float32)]
#6000 0.6908003 [array([[ 1.2806787 , -0.6452718 ],
#                       [ 0.5530104 ,  0.35553902]], dtype=float32), array([[ 0.4664025 ],
#                                                                           [-0.01079083]], dtype=float32)]
#7000 0.6904306 [array([[ 1.2923132 , -0.64527315],
#                       [ 0.590289  ,  0.3555404 ]], dtype=float32), array([[0.49329597],
#                                                                           [0.01083042]], dtype=float32)]
#8000 0.6900164 [array([[ 1.3053316 , -0.6460379 ],
#                       [ 0.6292683 ,  0.35701123]], dtype=float32), array([[0.522247  ],
#                                                                           [0.03224266]], dtype=float32)]
#9000 0.689549 [array([[ 1.3198922 , -0.64757645],
#                      [ 0.6700907 ,  0.3599293 ]], dtype=float32), array([[0.5533905 ],
#                                                                          [0.05356278]], dtype=float32)]
#10000 0.689018 [array([[ 1.336169  , -0.6499161 ],
#                       [ 0.71291447,  0.36428452]], dtype=float32), array([[0.5868867 ],
#                                                                           [0.07490435]], dtype=float32)]
#
#Hypothesis:  [[0.49029958]
#              [0.5034444 ]
#              [0.50753003]
#              [0.5121067 ]]
#Predicted(Y):  [[0.]
#                [1.]
#                [1.]
#                [1.]] 
#Accuracy:  0.75
