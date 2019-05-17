# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 5 P105 - Attention Example
import tensorflow as tf

bsZ=2
wSz=3
rnnSz=3

#Pretend Encoder Output
eo =( ((  1, 2, 3, 4 ),
       (  1, 1, 1, 1 ),
       (  1, 1, 1, 1 ),
       ( -1, 0,-1, 0)),
      ((  1, 2, 3, 4 ),
       (  1, 1, 1, 1 ),
       (  1, 1, 1, 1 ),
       ( -1, 0,-1, 0 )) )

#Pretend Encoder Out
encOut=tf.constant(eo,tf.float32)

#Pretend Attention - these weights could be learned
AT=( (.6,.25,.25),
     (.2,.25,.25),
     (.1,.25,.25),
     (.1,.25,.25) )

#Pretend Attention Weights
wAT=tf.constant(AT,tf.float32)

encAT=tf.tensordot(encOut,wAT,[[1],[0]])
sess=tf.Session()

print sess.run(encAT)

decAT=tf.transpose(encAT,[0,2,1])
print sess.run(decAT)

#P108 - GRU Example
#cell=tf.contrib.GRUCell(rnnSz)
#encOutSmall,encStateS=tf.nn.dynamic_rnn(cell,smallerEmbs,...)
#encOutLarge,encStateL=tf.nn.dynamic_rnn(cell,largerEmbs,...)
