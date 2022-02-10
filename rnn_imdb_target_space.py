# Example code using bespoke Keras layers to implemente a RNN in target space, with embedding layer.
# This code is to accompany the paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L. Citi, https://jmlr.org/papers/v23/20-040.html 
# Repository: https://github.com/mikefairbank/dlts_paper_code
# Note that the experiments in the paper used the TF1 version of this code (provided in the same repository)
# Please cite the above paper if this code or future variants of it are used in future academic work.
# 
#
# Example, to run code from command line, use either:
#python3 rnn_imdb_target_space.py  --weights  --mbs 40  --lr 0.001 --top_words 500 --max_review_length 50 --rsl 60 --max_epoch 10
#python3 rnn_imdb_target_space.py  --weights  --mbs 40  --lr 0.001 --top_words 5000 --max_review_length 500 --rsl 60 --max_epoch 10 --lstm 
#python3 rnn_imdb_target_space.py  --targets  --mbs 40 --rbs 40  --lr 0.001 --top_words 5000 --max_review_length 500 --rsl 60 --max_epoch 10
#python3 rnn_imdb_target_space.py  --targets   --mbs 40 --rbs 40 --lr 0.001 --top_words 500 --max_review_length 50 --rsl 60 --max_epoch 10
#python3 rnn_imdb_target_space.py  --targets  --mbs 40 --rbs 40 --lr 0.001 --top_words 50 --max_review_length 5 --rsl 6 --max_epoch 1
#python3 rnn_imdb_target_space.py  --weights  --mbs 40 --rbs 40 --lr 0.001 --top_words 50 --max_review_length 5 --rsl 6 --max_epoch 1

#python3 rnn_imdb_target_space.py  --targets  --mbs 40  --lr 0.001 --top_words 5000 --max_review_length 500 --rsl 60 --max_epoch 50 --log_results --steps_per_epoch 100


import sys
import argparse
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ts_layers as ts  # This contains the main target-space program logic
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--targets', action='store_true', help='Use target space')
parser.add_argument('--weights', action='store_true', help='Use weight space')
parser.add_argument('--sgd', action='store_true',  default=False, help='Use SGD not Adam')
parser.add_argument('--mbs', type=int, default=None, help='mini batch size (default = batch_size)')
parser.add_argument('--rbs', type=int, default=None, help='realisation batch size (default = batch_size)')
parser.add_argument('--lr', type=float, help='learning_rate')
parser.add_argument('--max_its', type=int, default=4000, help='max_its')
parser.add_argument('--max_epoch', type=int)
parser.add_argument('--sequence_length', type=int, default=None, help='sequence_length')
parser.add_argument('--rsl', type=int, help='realisation_sequence_length')
parser.add_argument('--lstm', action='store_true', help='lstm')
parser.add_argument('--pic', type=float, default=0.1, help='pseudoinverse_regularisation_constant')
parser.add_argument('--avoid_projection', action='store_true', help='avoid target initial projection')
parser.add_argument('--ti', type=float, default=1.0, help='targets initialiser magnitude')
parser.add_argument('--top_words', type=int, default=500)
parser.add_argument('--max_review_length', type=int, default=50)
parser.add_argument('--embedding_length', type=int, default=32)
parser.add_argument('--context_nodes', type=int, default=100)
parser.add_argument('--steps_per_epoch',type=int,default=100)
parser.add_argument('--log_results',action='store_true')

args = parser.parse_args()

#np.random.seed(7)
top_words = args.top_words
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)
max_review_length = args.max_review_length 
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)
X_train=X_train.reshape([-1,max_review_length])
X_test=X_test.reshape([-1,max_review_length])
y_train=y_train.reshape([-1])
y_test=y_test.reshape([-1])


embedding_vector_length = args.embedding_length
use_target_space=args.targets or not(args.weights)
learning_rate=args.lr if args.lr!=None else (0.1 if args.sgd else 0.001)
training_set_size=X_train.shape[0]
mini_batch_size=args.mbs if args.mbs!=None else training_set_size
max_its=args.max_its
if args.max_epoch !=None:
    max_its=(X_train.shape[1]*args.max_epoch)//mini_batch_size

seqLength=max_review_length
if args.rsl!=None:
    realisation_sequence_length=args.rsl
else:
    realisation_sequence_length=seqLength
do_initial_targets_projection=not args.avoid_projection
realisation_batch_size=args.rbs if args.rbs!=None else mini_batch_size
realisation_seq_length=args.rsl


num_inputs_per_loop=embedding_vector_length
num_outputs=2



use_target_space=not args.weights
if use_target_space:
    return_sequences=True
    class TSModel(keras.Model):
        def __init__(self):
            super(TSModel, self).__init__()
            self.fixed_targets_input_matrix=tf.constant(tf.random.uniform([realisation_batch_size,realisation_seq_length,embedding_vector_length],minval=-1, maxval=1.0, dtype=tf.float32))
            self.tslayers=[]
            
            self.tslayers.append(layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length,embeddings_initializer=tf.keras.initializers.TruncatedNormal(
mean=0.0, stddev=0.1)))
            if args.lstm:
                self.tslayers.append(ts.TSLSTM(args.context_nodes, seq_length=max_review_length, realisation_batch_size=realisation_batch_size, realisation_seq_length=realisation_seq_length, return_sequences=return_sequences, activation='tanh', pseudoinverse_l2_regularisation=args.pic))
            else:
                self.tslayers.append(ts.TSRNNDense(args.context_nodes, seq_length=max_review_length, realisation_batch_size=realisation_batch_size, realisation_seq_length=realisation_seq_length, return_sequences=return_sequences, activation='tanh', pseudoinverse_l2_regularisation=args.pic))

            if return_sequences:
                self.tslayers.append(ts.TSDense(num_outputs, realisation_batch_size=[realisation_batch_size,realisation_seq_length], pseudoinverse_l2_regularisation=args.pic))
                self.tslayers.append(layers.Lambda(lambda x: x[:,-1,:]))
            else:
                self.tslayers.append(ts.TSDense(num_outputs, realisation_batch_size=realisation_batch_size, pseudoinverse_l2_regularisation=args.pic))

            if not args.avoid_projection:
                self.initialise_target_layers_with_projection()
            
        def call(self, inputs):
            x=inputs
            x_targets=self.fixed_targets_input_matrix
            for l in self.tslayers:
                if isinstance(l, layers.Embedding):
                    x=l(x)
                elif isinstance(l, ts.TSLayer):
                    x_targets,x=l(x_targets,x)
                else:
                    x_targets,x=l(x_targets),l(x)
            return x
        
        def initialise_target_layers_with_projection(self):
            x=self.fixed_targets_input_matrix
            for l in self.tslayers:
                if isinstance(l, layers.Embedding):
                    x=x
                elif isinstance(l, ts.TSLayer):
                    x=l.initialise_targets_with_projection(x)        
                else:
                    x=l(x)
                
    keras_model=TSModel()
else:
    inputs = keras.Input(shape=(max_review_length), name='input')
    x = layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length)(inputs)
    if args.lstm:
        x = layers.LSTM(args.context_nodes, return_sequences=False,activation='tanh')(x)
    else:
        x = layers.SimpleRNN(args.context_nodes, return_sequences=False,activation='tanh')(x)
    outputs = layers.Dense(num_outputs, name='output')(x)
    keras_model = keras.Model(inputs=inputs, outputs=outputs)
    #keras_model.summary()

if args.sgd:
    optimizer=keras.optimizers.SGD(learning_rate)
else:
    optimizer=keras.optimizers.Adam(learning_rate)
keras_model.compile(optimizer=optimizer,  
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

if args.log_results:
    print("targets,iter,epoch,accuracy,test_accuracy,mbs,sl,rbs,rsl,top_words,embedding_vector_length,lr,use_lstm,steps_per_epoch,pic,cumul_cpu_time")
    start_time = time.time()
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            iteration=epoch*args.steps_per_epoch
            num_training_patterns_seen=iteration*mini_batch_size
            true_epoch=num_training_patterns_seen/training_set_size
            print(use_target_space,iteration,round(true_epoch,3),logs["sparse_categorical_accuracy"],logs["val_sparse_categorical_accuracy"],mini_batch_size,max_review_length, realisation_batch_size,args.rsl,top_words,embedding_vector_length,learning_rate,args.lstm,args.steps_per_epoch,args.pic,time.time()-start_time,sep=",")
    callbacks=CustomCallback()
    verbose=0
else:
    callbacks=None
    verbose=1
              
history = keras_model.fit(X_train, y_train,
                batch_size=mini_batch_size,
                epochs=(args.max_epoch*training_set_size+1)//(mini_batch_size*args.steps_per_epoch),
                validation_data=(X_test, y_test),validation_freq=1,
                verbose=verbose,steps_per_epoch=args.steps_per_epoch,callbacks=callbacks) 

