# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 6 Deep RL
import tensorflow as tf
import numpy as np

import gym
game=gym.make('FrozenLake-v0')
for i in range(1000):
    st=game.reset()
    for stps in range (99):
        act=np.random.randint(0,4)
        nst,rwd,dn,_=game.step(act)
        #Update T and # R:

        if dn: break


inptSt = tf.placeholder(dtype=tf.int32)
oneH=tf.one_hot(inptSt,16)
Q=tf.Variable(tf.random_uniform([16,4],0,0.01))
qVals=tf.matmul([oneH],Q)
outAct=tf.argmax(qVals,1)
