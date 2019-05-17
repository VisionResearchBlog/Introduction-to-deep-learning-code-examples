# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 2 - #PG 48 - Using layers for the 1st time and adding saved checkpoints
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data

#Model:
# Pr(A(x))=softmax( relu( xU+Ub )V + Vb )
#if you have saved checkpoint
pretrained=False
save_dir=os.getcwd() #not mentioned

mnist =  input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz=100

img=tf.placeholder(tf.float32,[batchSz,784])
ans=tf.placeholder(tf.float32,[batchSz,10])

#introduction of layers
L1Output=layers.fully_connected(img,756)
prbs=layers.fully_connected(L1Output,10,tf.nn.softmax)

#Cross entropy will be used for out loss function
xEnt=tf.reduce_mean(-tf.reduce_sum(ans*tf.log(prbs),reduction_indices=[1]))

#learning rate should be between 0.01-0.05
train=tf.train.GradientDescentOptimizer(0.05).minimize(xEnt)
numCorrect=tf.equal(tf.argmax(prbs,1),tf.argmax(ans,1))
accuracy=tf.reduce_mean(tf.cast(numCorrect,tf.float32))

sess=tf.Session()
save0b=tf.train.Saver()

if pretrained:
    save0b.restore(sess,save_dir+"/mylatest.ckpt")
else:
    sess.run(tf.global_variables_initializer())

epochs=100
#training loop
for i in range(epochs):
    imgs,anss=mnist.train.next_batch(batchSz)
    acc,ignore=sess.run([accuracy, train],feed_dict={img: imgs, ans: anss}) #revision pg 37
    print "Train Accuracy: %r" % (acc)


#Test loop on novel data outsidwe training
sumAcc=0
for i in range(epochs):
    imgs,anss=mnist.test.next_batch(batchSz)
    sumAcc+=sess.run(accuracy,feed_dict={img: imgs, ans: anss})
print "Test Accuracy: %r" % (sumAcc/epochs)

save0b.save(sess,save_dir+"/mylatest.ckpt")
