# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================

#Simple example of tensorflow graph and session

# =============================================================================
#CHAPTER 2 - PG 32-33


import tensorflow as tf

#PG 32 - TODO

#PG33
img=tf.placeholder(tf.float32,shape=[28,28]) #example reserving item in memory

bt=tf.random_normal([10],stddev=.1)
bias=tf.Variable(bt)
weights=tf.Variable(tf.random_normal([784,10],stddev=.1))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(bias))
print(sess.run(weights))
# =============================================================================
