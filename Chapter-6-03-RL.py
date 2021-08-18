# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 6 Deep RL - PG123
import tensorflow as tf
import numpy as np
import gym

#This example gives some code sketches that need to be used with the pseudocode
#on PG128 Figure 6.11

#inptSt = tf.placeholder(dtype=tf.int32)
#oneH=tf.one_hot(inptSt,16)
#Q=tf.Variable(tf.random_uniform([16,4],0,0.01))
#qVals=tf.matmul([oneH],Q)
#outAct=tf.argmax(qVals,1)

#P127 Figure 6.12 - Using expected discounted reward
state=tf.placeholder(shape=[None,4],dtype=tf.float32)
W=tf.Variable(tf.random_uniform([4,8],dtype=tf.float32))
hidden=tf.nn.relu(tf.matmul(state,W))
O=tf.Variable(tf.random_uniform([8,2],dtype=tf.float32))
output=tf.nn.softmax(tf.matmul(hidden,0))

rewards=tf.placeholder(shape=[None],dtype=tfloat32)
actions=tf.placeholder(shape=[None],dtype=tfloat32)
indices=tf.range(0,tf.shape(output)[0])*2 + actions
actProbs=tf.gather(tf.reshape(output, [-1]), indices)
aloss=-tf.reduce_mean(tf.log(actProbs)*rewards)
trainOp=tf.train.AdamOptimizer(.01).minimize(aloss)

#P132 Figure 6.13 - Using Actor/Critic Value
V1=tf.Variable(tf.random_normal([4,8],dtype=tf.float32,stddev=.1))
v1Out=tf.nn.relu(tf.matmul(state,V1))
V2=tf.Variable(tf.random_normal([8,1],dtype=tf.float32,stddev=.1))
v2Out=tf.matmul(V1Out,V2)
advantage=rewards-v2Out
aLoss=-tf.reduce_mean(tf.log(actProbs)*advantage)
cLoss=tf.reduce_mean(tf.square(rewards-vOut))
loss=aLoss+cLoss

#nextQ=tf.placeholder(shape=[1,4],dtype=tf.float32)
#loss=tf.reduce_sum(tf.square(nextQ-qVals))
#trainer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
#updateMod=trainer.minimize(loss)
#init=tf.global_variables_initializer()

gamma=.99
rTot=0
numTrials=2000

env = gym.make('CartPole-v0')
#env.reset()
#general idea is below from: https://gym.openai.com/docs/#environments
#for _ in range(numTrials):
#    env.render()
#    env.step(env.action_space.sample()) # take a random action
#env.close()

# Exercise update below using ideas from P127

# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(numTrials):
#         e=50.0/(i+50)
#         s=game.reset()
#         for j in range (99):
#             nActs,nxtQ=sess.run([outAct,qVals],feed_dict={inptSt: s})
#             nActsChoice=nActs #Choosing only first action (book typo?)
#
#             if np.random.rand(1)<e:
#                 nActsChoice=game.action_space.sample()
#             s1,rwd,dn,_=game.step(nActsChoice)
#             Q1=sess.run(qVals,feed_dict={inptSt:s1})
#             nxtQ[0,nActsChoice]=rwd + gamma*(np.max(Q1))
#             sess.run(updateMod,feed_dict={inptSt:s, nextQ:nxtQ})
#             rTot+=rwd
#             if dn: break
#             s=s1
#
# print"Percent games successful: ", rTot/numTrials
