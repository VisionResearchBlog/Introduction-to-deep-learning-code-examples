# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
#CHAPTER 7 - PG154 - GAN Example - learn the mean of a normal distribution
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

#Our target 'real' distribution - the generator network needs to approximate
#this will also form the basis for the discrimnators knowledge of 'real' data
target_mean=5
target_std=0.5

bSz,hSz,numStps,logEvery,genRange=8,4,5000,500,8
def log(x): return tf.log(tf.maximum(x,1e-5))

with tf.variable_scope('GEN'):
    gIn=tf.placeholder(tf.float32,shape=(bSz,1))
    g0=layers.fully_connected(gIn,hSz,tf.nn.softplus)
    G=layers.fully_connected(g0,1,None)
gParams=tf.trainable_variables()

def discriminator(input):
    h0=layers.fully_connected(input,hSz*2,tf.nn.relu)
    h1=layers.fully_connected(h0,hSz*2,tf.nn.relu)
    h2=layers.fully_connected(h1,hSz*2,tf.nn.relu)
    h3=layers.fully_connected(h2,1,tf.sigmoid)
    return h3

dIn=tf.placeholder(tf.float32,shape=(bSz,1))
with tf.variable_scope('DIS'):
    D1=discriminator(dIn)
with tf.variable_scope('DIS',reuse=True):
    D2=discriminator(G)
dParams=[v for v in tf.trainable_variables()
                if v.name.startswith('DIS')]

#Generator loss increases as accuracy on real images decreases
gLoss=tf.reduce_mean(-log(D2))

#Discrimator loss increases as accuracy on real images decreases and fake images
#increases, i.e. increased making of false negatives and false positives
dLoss=0.5*tf.reduce_mean(-log(D1) -log(1-D2))
gTrain=tf.train.AdamOptimizer(.001).minimize(gLoss, var_list=gParams)
dTrain=tf.train.AdamOptimizer(.001).minimize(dLoss, var_list=dParams)

sess=tf.Session()
sess.run(tf.global_variables_initializer())


gmus,gstds=[],[]
for i in range(numStps+1):
    real=np.random.normal(target_mean,target_std,(bSz,1))
    fakeRnd=np.random.uniform(-genRange,genRange,(bSz,1))

    #update discriminator
    lossd,gout,_=sess.run([dLoss,G,dTrain],{gIn:fakeRnd,dIn:real})
    gmus.append(np.mean(gout))
    gstds.append(np.std(gout))

    #update generator
    fakeRnd=np.random.uniform(-genRange,genRange,(bSz,1))
    lossg,_=sess.run([gLoss,gTrain],{gIn:fakeRnd})
    if i % logEvery == 0:
        frm=np.max(i-5,0)
        cmu=np.mean(gmus[frm: (i+1)])
        cstd=np.mean(gstds[frm: (i+1)])
        print('{}:\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.
            format(i,lossd,lossg,cmu,cstd))
