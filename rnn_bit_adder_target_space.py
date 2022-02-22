# Example code using bespoke Keras layers to implemente a recurrent NN in target space.
# This code is to accompany the paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L. Citi, https://jmlr.org/papers/v23/20-040.html 
# Repository: https://github.com/mikefairbank/dlts_paper_code
# Note that the experiments in the paper used the TF1 version of this code (provided in the same repository)
# Please cite the above paper if this code or future variants of it are used in future academic work.

#
# Example, to run code from command line, use:
# python3 rnn_bit_adder_target_space.py --targets --delay_length 60 --rsl 50 --adder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ts_layers as ts
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--targets', action='store_true', default=True, help='Use target space')
parser.add_argument('--weights', action='store_true', help='Use weight space')
parser.add_argument('--sgd', action='store_true',  default=False, help='Use SGD (instead of Adam)')
parser.add_argument('--training_set_size', type=int, default=8000, help='training set size')
parser.add_argument('--mbs', type=int, default=100, help='mini batch size (default = batch_size)')
parser.add_argument('--rbs', type=int, default=None, help='realisation batch size (default = batch_size)')
parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
#parser.add_argument('--max_its', type=int, default=4000, help='max_its')
parser.add_argument('--delay_length', type=int, default=20, help='delay_length')
parser.add_argument('--sequence_length', type=int, default=None, help='sequence_length')
parser.add_argument('--rsl', type=int, help='realisation_sequence_length')
parser.add_argument('--adder', action='store_true', help='bit adder')
#parser.add_argument('--lstm', action='store_true', help='lstm') # Not implmented here.
parser.add_argument('--pic', type=float, default=0.1, help='pseudoinverse_regularisation_constant')
parser.add_argument('--avoid_projection', action='store_true', help='avoid target initial projection')
#parser.add_argument('--ti', type=float, default=1.0, help='targets initialiser magnitude')
args = parser.parse_args()

def calculateInputOutputSequences(seqLength, batchSize, delay_length, useXor, seed1, bitAdder):
    # This function generates the bit-sequence manipulation tasks described in the RNN experiments of the paper.
    np.random.seed(seed1)
    randomBitSequence=np.random.randint(2, size=(batchSize,seqLength,1))
    yTargets=randomBitSequence[:,0:-delay_length,:]  # removes the last "delay_length" bits from the bit sequence (since these cannot possibly be predicted)
    if useXor:
        a=yTargets
        b=randomBitSequence[:,delay_length:seqLength,:]
        yTargets=np.bitwise_xor(a,b)
    elif bitAdder:
        a=yTargets
        b=randomBitSequence[:,delay_length:seqLength,:]
        carryBits=np.zeros_like(b[:,0,:])
        results_list=[]
        for time_step in range(b.shape[1]):
            a_col=a[:,time_step,:]
            b_col=b[:,time_step,:]
            sum1=a_col+b_col+carryBits
            carryBits=sum1//2    
            sum1=sum1%2
            results_list.append(sum1)
        yTargets=np.stack(results_list, axis=1)
    return [randomBitSequence.astype(np.float32),yTargets.astype(np.float32)]


delay_length=args.delay_length
seq_length=args.sequence_length if args.sequence_length!=None else delay_length+50
if seq_length<=delay_length:
    raise Exception("Illegal Argument, seq_length<delay_length")
adder=args.adder
train_inputs,train_labels=calculateInputOutputSequences(seq_length, args.training_set_size, delay_length, useXor=False, seed1=1, bitAdder=adder)
test_inputs,test_labels=calculateInputOutputSequences(seq_length, min(args.training_set_size,1000), delay_length, useXor=False, seed1=2, bitAdder=adder)
print("train data shape, input",train_inputs.shape,"output",train_labels.shape)
input_vector_length=1
num_hids=delay_length+(5 if adder else 3)
num_output_categories=2
num_sequence_outputs_to_retain=seq_length-delay_length
batch_size=args.mbs
epochs=50000*batch_size//args.training_set_size*4
    
use_target_space=not args.weights
if use_target_space:
    realisation_batch_size=args.rbs if args.rbs!=None else batch_size
    realisation_seq_length=args.rsl
    class TSModel(keras.Model):
        def __init__(self):
            super(TSModel, self).__init__()
            self.fixed_targets_input_matrix=tf.constant(train_inputs[:realisation_batch_size,:realisation_seq_length,:],tf.float32)
            self.tslayers=[]
            
            self.tslayers.append(ts.TSRNNDense(num_hids, seq_length=seq_length, realisation_batch_size=realisation_batch_size, realisation_seq_length=realisation_seq_length, return_sequences=True, activation='tanh', pseudoinverse_l2_regularisation=args.pic))
            self.tslayers.append(ts.TSDense(num_output_categories, realisation_batch_size=[realisation_batch_size,realisation_seq_length], pseudoinverse_l2_regularisation=args.pic))
            if num_sequence_outputs_to_retain<seq_length:
                self.tslayers.append(layers.Lambda(lambda x: x[:,-num_sequence_outputs_to_retain:,:]))
            if not args.avoid_projection:
                self.initialise_target_layers_with_projection()
            
        def call(self, inputs):
            x=inputs
            x_targets=self.fixed_targets_input_matrix
            for l in self.tslayers:
                if isinstance(l, ts.TSLayer):
                    x_targets,x=l([x_targets,x])
                else:
                    x_targets,x=l(x_targets),l(x)
            return x
        
        def initialise_target_layers_with_projection(self):
            x=self.fixed_targets_input_matrix
            for l in self.tslayers:
                if isinstance(l, ts.TSLayer):
                    x=l.initialise_targets_with_projection(x)        
                else:
                    x=l(x)
                
                
    keras_model=TSModel()
else:
    inputs = keras.Input(shape=(seq_length,input_vector_length), name='input')
    x = layers.SimpleRNN(num_hids, return_sequences=True,activation='tanh')(inputs)
    x=x[:,-num_sequence_outputs_to_retain:,:]
    outputs = layers.Dense(num_output_categories, name='output')(x)
    keras_model = keras.Model(inputs=inputs, outputs=outputs)

#keras_model.summary()

if args.sgd:
    optimizer=keras.optimizers.SGD(learning_rate)
else:
    optimizer=keras.optimizers.Adam(args.lr)

keras_model.compile(optimizer=optimizer,  
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
history = keras_model.fit(train_inputs, train_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(test_inputs, test_labels),validation_freq=5,
                verbose=1)

