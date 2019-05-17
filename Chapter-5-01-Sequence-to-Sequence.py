# =============================================================================
# Examples from Eugene Charniak's Introduction to Deep Learning 2018 MIT Press
# =============================================================================
 #CHAPTER 5

#Code fragments to give general idea of sequence to sequence network
#Exercise is left to reader to implement
import tensorflow as tf

vfSz=10
embedSz=10

with tf.variable_scope("enc"):
    F=tf.Variable(tf.random_normal((vfSz,embedSz),stddev=.1))
    embs=tf.nn.embedding_lookup(F,encIn)
    embs=tf.nn.dropout(embs,keepPrb)
    cell=tf.contrib.rnn.GRUCell(rnnSz)
    initState=cell.zero_state(bSz,tf.float32)
    encOut,encState=tf.nn.dynamic_rnn(cell,embs,initial_state=initState)

with tf.variable_scope("dec"):
    E=tf.Variable(tf.random_normal((veSz,embedSz),stddev=.1))
    embs=tf.nn.embedding_lookup(E,decIn)
    embs=tf.nn.dropout(embs,keepPrb)
    cell=tf.contrib.rnn.GRUCell(rnnSz)
    decOut,_=tf.nn.dynamic_rnn(cell,embs,initial_state=encState)


W=tf.Variable(tf.random_normal([rnnSz,veSz],stddev=.1))
b=tf.Variable(tf.random_normal([veSz],stddev=.1))
logits=tf.tensordot(decOut,W,axes=[[2],[0]])+b
loss=tf.contrib.seq2seq.sequence_loss(logits, ans, tf.ones([bSz,wSz]))

# =============================================================================
