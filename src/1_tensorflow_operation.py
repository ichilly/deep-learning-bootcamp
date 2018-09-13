import tensorflow as tf

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
adder_node = node1 + node2  # build graph with tensors

print(adder_node)
# Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run(adder_node)) # run graph with operation
# 7.0
