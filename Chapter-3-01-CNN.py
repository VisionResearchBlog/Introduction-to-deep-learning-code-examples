# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 2 - PG 34
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist =  input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz=100
#W=tf.Variable(tf.random_normal([784,10],stddev=.1))
W=tf.Variable(tf.random_normal([1568,10],stddev=0.1))
b=tf.random_normal([10],stddev=.1)

img=tf.placeholder(tf.float32,[batchSz,784])
image=tf.reshape(img,[100,28,28,1])

flts=tf.Variable(tf.random_normal([4,4,1,16],stddev=0.1))
convOut=tf.nn.conv2d(image,flts,[1,2,2,1],"SAME")
bias=tf.Variable(tf.zeros([16]))
convOut+=bias
convOut=tf.nn.relu(convOut)
flts2=tf.Variable(tf.random_normal([2,2,16,32],stddev=0.1))
convOut2=tf.nn.conv2d(convOut,flts2,[1,2,2,1],"SAME")
convOut2=tf.reshape(convOut2,[100,1568])

ans=tf.placeholder(tf.float32,[batchSz,10])

#prbs=tf.nn.softmax(tf.matmul(img,W)+b)
prbs=tf.nn.softmax(tf.matmul(convOut2,W)+b)

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
