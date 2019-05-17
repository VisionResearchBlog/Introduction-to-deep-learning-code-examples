# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 6 Deep RL - PG123
import tensorflow as tf
import numpy as np
import gym

inptSt = tf.placeholder(dtype=tf.int32)
oneH=tf.one_hot(inptSt,16)
Q=tf.Variable(tf.random_uniform([16,4],0,0.01))
qVals=tf.matmul([oneH],Q)
outAct=tf.argmax(qVals,1)

nextQ=tf.placeholder(shape=[1,4],dtype=tf.float32)
loss=tf.reduce_sum(tf.square(nextQ-qVals))
trainer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateMod=trainer.minimize(loss)
init=tf.global_variables_initializer()

gamma=.99
game=gym.make('FrozenLake-v0')
rTot=0

numTrials=2000

with tf.Session() as sess:
    sess.run(init)
    for i in range(numTrials):
        e=50.0/(i+50)
        s=game.reset()
        for j in range (99):
            nActs,nxtQ=sess.run([outAct,qVals],feed_dict={inptSt: s})
            nActsChoice=nActs[0] #Choosing only first action (book typo?)

            if np.random.rand(1)<e:
                nActsChoice=game.action_space.sample()
            s1,rwd,dn,_=game.step(nActsChoice)
            Q1=sess.run(qVals,feed_dict={inptSt:s1})
            nxtQ[0,nActsChoice]=rwd + gamma*(np.max(Q1))
            sess.run(updateMod,feed_dict={inptSt:s, nextQ:nxtQ})
            rTot+=rwd
            if dn: break
            s=s1

print"Percent games successful: ", rTot/numTrials
