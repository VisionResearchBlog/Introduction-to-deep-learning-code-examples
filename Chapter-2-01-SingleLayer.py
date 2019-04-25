# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 2 - PG 34
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist =  input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz=100
W=tf.Variable(tf.random_normal([784,10],stddev=.1))
b=tf.random_normal([10],stddev=.1)
 
img=tf.placeholder(tf.float32,[batchSz,784])
ans=tf.placeholder(tf.float32,[batchSz,10])
 
prbs=tf.nn.softmax(tf.matmul(img,W)+b)
#Cross entropy will be used for out loss function
xEnt=tf.reduce_mean(-tf.reduce_sum(ans*tf.log(prbs),reduction_indices=[1]))

train=tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
numCorrect=tf.equal(tf.argmax(prbs,1),tf.argmax(ans,1))
accuracy=tf.reduce_mean(tf.cast(numCorrect,tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    imgs,anss=mnist.train.next_batch(batchSz)
    #sess.run(train,feed_dict={img: imgs, ans: anss}) #initial version
    #revision pg 37 outputs acc so we can have acc data print as we train
    acc,ignore=sess.run([accuracy, train],feed_dict={img: imgs, ans: anss}) 
    print "Train Accuracy: %r" % (acc)

sumAcc=0
for i in range(1000):
    imgs,anss=mnist.test.next_batch(batchSz)
    sumAcc+=sess.run(accuracy,feed_dict={img: imgs, ans: anss})
print "Test Accuracy: %r" % (sumAcc/1000)

# =============================================================================
