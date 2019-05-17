# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
#CHAPTER 7 - PG142 - Convolutional Autoencoder Example
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data

mnist =  input_data.read_data_sets("MNIST_data/", one_hot=True)
batchSz=100

orgI=tf.placeholder(tf.float32,shape=[None,784])
I=tf.reshape(orgI,[-1,28,28,1])
smallI=tf.nn.max_pool(I,[1,2,2,1],[1,2,2,1],"SAME")
smallerI=tf.nn.max_pool(smallI,[1,2,2,1],[1,2,2,1],"SAME")
feat=tf.Variable(tf.random_normal([2,2,1,1],stddev=.1))
recon=tf.nn.conv2d_transpose(smallerI,feat,[100,14,14,1],[1,2,2,1],"SAME")

loss=tf.reduce_sum(tf.square(recon-smallI))
trainop=tf.train.AdamOptimizer(.0003).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
epochs=8001

for i in range(epochs):
    batch=mnist.train.next_batch(batchSz)
    fd={orgI:batch[0]}
    oo,ls,ii,_=sess.run([smallI,loss,recon,trainop],fd)
    lossAccum[i]=ls
    print ls
