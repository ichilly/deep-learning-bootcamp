import tensorflow as tf

p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
adder_node = p1 + p2

sess = tf.Session()
print(sess.run(adder_node, feed_dict={p1: 3, p2: 4.5}))
# 7.5
