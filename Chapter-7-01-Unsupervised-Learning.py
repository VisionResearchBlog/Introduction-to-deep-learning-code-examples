# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 7 - PG140 - Autoencoder Examples

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
mnist =  input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz=100
img=tf.placeholder(tf.float32,[batchSz,784])

#encoder
E1=layers.fully_connected(img,256,tf.nn.sigmoid)
E2=layers.fully_connected(E1,128,tf.nn.sigmoid)
#decoder
D2=layers.fully_connected(E2,256,tf.nn.sigmoid)
D1=layers.fully_connected(D2,256,tf.nn.softmax)

Output=layers.fully_connected(D1,784)
loss=tf.reduce_sum(tf.square(img-Output))
train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
epochs=8000

for i in range(epochs):
    batch=mnist.train.next_batch(batchSz)
    fd={img:batch[0]}
    ls,_=sess.run([loss,train],fd)
    print ls
