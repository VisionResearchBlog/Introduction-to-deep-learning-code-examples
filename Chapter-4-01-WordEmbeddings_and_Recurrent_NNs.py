# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 4 - No full working code displayed - This chapter sets up concepts
 #for creating an rnn

import tensorflow as tf

#no data set suggested

batchSz=100 #inferred from text
vocabSz=7500
embedSz=30

#P76
inpt=tf.placeholder(tf.int32,shape=[batchSz])
answr=tf.placeholder(tf.int32,shape=[batchSz])
E=tf.Variable(tf.random_normal([vocabSz, embedSz], stddev = 0.1))
embed=tf.nn.embedding_lookup(E,inpt)

#P77
xEnt=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=answr)
loss=tf.reduce_sum(xEnt)

#P79
embed2=tf.nn.embedding_lookup(E,inpt2)
both=tf.concat([embed,embed2],1)

#P81
#placeholder means we can do dropout during training but switch off in Test
keepP=tf.placeholder(tf.float32)
w1Out=tf.nn.dropout(w1Out,keepP)

#P82
#L2 regularization alpha=0.01, weight on L2
loss=tf.reduce_sum(xEnt) + .01*tf.nn.l2_loss(W1)

#P86
rnn=tf.contrib.rnn.BasicRNNCell(rnnSz)
rnn=tf.contrib.rnn.LSTMCell(rnnSz) #P91
initialState=rnn.zero_state(batchSz,tf.float32)
outputs, nextState = tf.nn.dynamic_rnn(rnn, embeddings, initial_state=initialState)

#P88
output2 = tf.reshape(output,[batchSz*windowSz,rnnSz])
logits = matmul(output2,W)
tf.tensordot(outputs,W,[[2],[0]])
inputSt=sess.run(initialSt)
for i in range(numExamps):
    "read in words and embed them"
    logts,nxts=sess.run([logits,nextState],{input=wrds, nextState=inputSt})

# =============================================================================
