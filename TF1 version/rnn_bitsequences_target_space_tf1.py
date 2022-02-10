# This code is to accompany the paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L. Citi, https://jmlr.org/papers/v23/20-040.html 
# This example code uses TF1 code, to replicate the bit-sequence memorisation and addition RNN experiments from the paper.  
# Please cite the above paper if this code or future variants of it are used in future academic work.

# Note that this LSTM implementation is different from the one used in the paper's experiments.  The paper used a standard 
# Tensorflow library to build the LSTM.  Since that publication this work was extended as a prototype to implement LSTM in target space.

#Example: 
# python3 rnn_bitsequences_target_space_tf1.py --targets --adam --delay_length 90 --sequence_length 170 --batch_size 400 --max_its 25000 --scu --pic 0.1
# python3 rnn_bitsequences_target_space_tf1.py --weights --lstm --adam --pic 0.1 --batch_size 8000 --rbs 100 --mbs 100 --max_its 50000 --lr 0.001 --adder --delay_length 40
import sys
from enum import Enum
import argparse
import time
import math
import tensorflow as tf
import numpy as np
if float(tf.__version__[0])>=2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--targets', action='store_true', help='Use target space')
parser.add_argument('--weights', action='store_true', help='Use weight space')
parser.add_argument('--adam', action='store_true',  default=False, help='Use Adam')
parser.add_argument('--batch_size', type=int, default=40, help='batch_size')
parser.add_argument('--mbs', type=int, default=None, help='mini batch size (default = batch_size)')
parser.add_argument('--rbs', type=int, default=None, help='realisation batch size (default = batch_size)')
parser.add_argument('--lr', type=float, help='learning_rate')
parser.add_argument('--max_its', type=int, default=4000, help='max_its')
parser.add_argument('--af', type=str, default="tanh", help='activation function (tanh/relu)')
parser.add_argument('--delay_length', type=int, default=2, help='delay_length')
parser.add_argument('--sequence_length', type=int, default=None, help='sequence_length')
parser.add_argument('--rsl', type=int, help='realisation_sequence_length')
parser.add_argument('--xor', action='store_true', help='xor')
parser.add_argument('--adder', action='store_true', help='bit adder')
parser.add_argument('--scu', action='store_true', help='SCU/OCU hybrid')
parser.add_argument('--lstm', action='store_true', help='lstm')
parser.add_argument('--pic', type=float, default=0.001, help='pseudoinverse_regularisation_constant')
parser.add_argument('--avoid_projection', action='store_true', help='avoid target initial projection')
parser.add_argument('--ti', type=float, default=1.0, help='targets initialiser magnitude')


args = parser.parse_args()


use_target_space=args.targets or not(args.weights)
use_adam=args.adam
learning_rate=args.lr if args.lr!=None else ((0.01 if use_target_space else 0.001) if use_adam  else 0.1)
max_its=args.max_its
batchSize=args.batch_size
mini_batch_size=args.mbs if args.mbs!=None else batchSize
realisation_batch_size=args.rbs if args.rbs!=None else mini_batch_size
afs=args.af
use_lstm=args.lstm
if args.xor and args.adder:
    print("xor and adder not compatible")
    sys.exit(0)

delay_length=args.delay_length
seqLength=args.sequence_length if args.sequence_length!=None else delay_length+50
if args.rsl!=None:
    realisation_sequence_length=args.rsl
else:
    realisation_sequence_length=seqLength
useXor=args.xor
bitAdder=args.adder
do_initial_targets_projection=not args.avoid_projection
use_cross_entropy=True
use_shortcuts=False


num_inputs_per_loop=1
num_outputs=2 if use_cross_entropy else 1
numContextNodes=delay_length+3 # this should be enough hidden nodes to solve this problem
if useXor or bitAdder:
    numContextNodes+=2 # xor problem is harder variant, so give the NN two more hidden nodes to benefit from.
hids=[num_inputs_per_loop, numContextNodes,num_outputs]
context_layer=1
datatype_tf=tf.float32 
targets_initialiser=args.ti


pseudoinverse_regularisation_constant=args.pic
af=tf.nn.tanh if afs=="tanh" else (tf.nn.relu if afs=="relu" else "")
weight_initialiser="GlorotNormal6" if afs=="tanh" else "HeNormal"

RealisationModes = Enum('RealisationModes', 'SCU OCU')
realisation_mode=RealisationModes.SCU if args.scu else RealisationModes.OCU
if args.scu and not use_target_space:
    print("Illegal argument, SCU with weight space")
    sys.exit(0)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

    
def computeLayerWeightMatrix(target_matrix, input_matrix, regularization, name):
    weight_matrix=tf.linalg.lstsq( input_matrix,target_matrix,l2_regularizer=regularization,fast=True)
    return weight_matrix

def next_batch(num, data, labels):
    idx = np.arange(0 , data.shape[1])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[:,idx,:]
    labels_shuffle = labels[:,idx]
    return data_shuffle, labels_shuffle

def initialiser_standard_deviation(n_in, n_out):
    return math.sqrt((6.0 if weight_initialiser=="GlorotNormal6" else 2.0)/(n_in+(n_out if weight_initialiser=="GlorotNormal6" else 0)))


def calculateInputOutputSequences(seqLength, batchSize, delay_length, useXor, seed1, bitAdder):
    np.random.seed(seed1)
    randomBitSequence=np.random.randint(2, size=(seqLength,batchSize,1))
    yTargets=randomBitSequence[0:seqLength-delay_length,:,:]  # removes the last "delay_length" bits from the bit sequence (since these cannot possibly be predicted)

    if useXor:
        a=yTargets
        b=randomBitSequence[delay_length:seqLength,:,:]
        yTargets=np.bitwise_xor(a,b)
        #print("randomBitSequence",randomBitSequence, "yTargets", yTargets)
        #sys.exit(0)
    elif bitAdder:
        a=yTargets
        b=randomBitSequence[delay_length:seqLength,:,:]
        carryBits=np.zeros_like(b[0,:,:])
        results_list=[]
        for time_step in range(b.shape[0]):
            a_col=a[time_step,:,:]
            b_col=b[time_step,:,:]
            sum1=a_col+b_col+carryBits
            carryBits=sum1//2    
            sum1=sum1%2
            results_list.append(sum1)
        yTargets=np.stack(results_list, axis=0)
        #print("a",a)
        #print("b",b)
        #print("sum",yTargets)
        #sys.exit(0)
    return [randomBitSequence.astype(np.float32),yTargets.astype(np.float32)]


def build_weight_matrices(hids,use_lstm):
    num_bias_inputs=1+numContextNodes+num_inputs_per_loop
    calculated_weight_matrices=[]
    for layer in range(len(hids)-1):
        num_nodes=hids[layer+1]
        if layer+1==context_layer and use_lstm:
            num_nodes*=4
        WL=tf.Variable(tf.truncated_normal([num_bias_inputs,num_nodes], stddev=initialiser_standard_deviation(num_bias_inputs, hids[layer+1]), dtype=datatype_tf, name="WeightMatrix"+str(layer))) #, seed=layer*2+9))
        if use_shortcuts:
            num_bias_inputs+=hids[layer+1]
        else:
            num_bias_inputs=1+hids[layer+1]
        calculated_weight_matrices.append(WL)
    return calculated_weight_matrices

def build_target_matrices(input_matrix,hids,use_lstm,mini_batch_size, seqLength):
    layer_target_matrices=[]
    # this was an incredibly lazy way to find the dimensions of the target matrices....
    temp_weight_matrices=build_weight_matrices(hids, use_lstm)
    [_, layer_sums_temp, _, unrolled_layers_corresponding_to_layers]=build_rnn(input_matrix, hids, temp_weight_matrices, None, mini_batch_size, seqLength)
    for layer in range(len(layer_sums_temp)):
        SL=layer_sums_temp[layer]
        TL=tf.Variable(tf.truncated_normal(SL.get_shape(), stddev=targets_initialiser, dtype=datatype_tf, name="TargetsMatrix"+str(layer))) 
        layer_target_matrices.append(TL)
    #print("Built target matrices",layer_target_matrices)
    return layer_target_matrices

def calculate_weight_matrices_from_targets(layer_target_matrices, realisation_mode, input_matrix, hids, use_lstm, mini_batch_size, seqLength):
    predicted_sums_from_targets=layer_target_matrices
    temp_weight_matrices=build_weight_matrices(hids, use_lstm)
    # push the current targets through the network, and force them to become the "sums", so that the output activations of every layer can be approximated
    # and so that the variable layer_inputs_temp can be found
    [_, layer_sums_temp, layer_inputs_temp, unrolled_layers_corresponding_to_layers_temp]=build_rnn(input_matrix, hids, temp_weight_matrices, predicted_sums_from_targets, mini_batch_size, seqLength)

    #build_rnn(input_matrix, hids, weight_matrices, given_loop_sums, mini_batch_size, seqLength)
    calculated_weight_matrices=[]
    if realisation_mode==RealisationModes.OCU:
        for layer in range(len(hids)-1):
            #print("layer",layer)
            #print("layer inputs",[layer_inputs[x] for x in unrolled_layers_corresponding_to_layers[layer]])
            BL_concatenated=tf.concat([layer_inputs_temp[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
            TL_concatenated=tf.concat([layer_target_matrices[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
            WL=computeLayerWeightMatrix(TL_concatenated, BL_concatenated, pseudoinverse_regularisation_constant, "CalculateWeightMatrix"+str(layer))
            calculated_weight_matrices.append(WL)
    elif realisation_mode==RealisationModes.SCU:
        # SCU + OCU hybrid
        #First layer uses OCU:
        layer=0
        BL_concatenated=tf.concat([layer_inputs_temp[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
        TL_concatenated=tf.concat([layer_target_matrices[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
        WL=computeLayerWeightMatrix(TL_concatenated, BL_concatenated, pseudoinverse_regularisation_constant, "CalculateWeightMatrix"+str(layer))
        calculated_weight_matrices.append(WL)

        temp_weight_matrices[0:len(calculated_weight_matrices)]=calculated_weight_matrices
        if context_layer!=1:
            print("Not implemented yet2")
            sys.exit(0)
        [_, layer_sums_temp, layer_inputs_temp, unrolled_layers_corresponding_to_layers_temp]=build_rnn(input_matrix, hids, temp_weight_matrices, None, mini_batch_size, seqLength)
        layer=1
        BL_concatenated=tf.concat([layer_inputs_temp[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
        TL_concatenated=tf.concat([layer_target_matrices[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
        WL=computeLayerWeightMatrix(TL_concatenated, BL_concatenated, pseudoinverse_regularisation_constant, "CalculateWeightMatrix"+str(layer))
        calculated_weight_matrices.append(WL)
        if layer!=len(hids)-2:
            print("Not implemented yet3")
            sys.exit(0)
    return calculated_weight_matrices

def build_rnn(input_matrix, hids, weight_matrices, given_loop_sums, mini_batch_size, seqLength):
    # Build recurrent neural network....
    initialHiddenState=tf.constant(0., shape=[mini_batch_size, numContextNodes], dtype=datatype_tf) 
    if use_lstm:
        initialHiddenCState=tf.fill(dims=[tf.shape(input_matrix)[1],numContextNodes], value=0.0)
        previous_context_layerC=initialHiddenCState
    previous_context_layer=initialHiddenState # This must not be tf.Variable
    unrolled_layers_corresponding_to_layers=[[] for _ in range(len(hids)-1)]
    bias_nodes=tf.ones_like(input_matrix[0,:,0:1])
    layer_sums=[]
    layer_inputs=[]
    network_output_matrices=[]
    for loop in range(seqLength):
        previous_layer_output=tf.concat([input_matrix[loop,:,:],previous_context_layer],axis=1)
        bias_nodes1=bias_nodes
        for layer in range(len(hids)-1):
            BL=tf.concat([bias_nodes1, previous_layer_output],axis=1, name="InputMatrix"+str(layer)+"-"+str(loop))    
            WL=weight_matrices[layer]
            SL=tf.matmul(BL,WL,name="Sums"+str(layer)+"-"+str(loop))
            layer_sums.append(SL)
            layer_inputs.append(BL)
            if given_loop_sums!=None:    
                SL=given_loop_sums[len(layer_sums)-1]
            if layer+1==context_layer and use_lstm:
                i, j, f, o = tf.split(value=SL, num_or_size_splits=4, axis=1)
                forget_bias_tensor = 1.0
                new_c = tf.multiply(previous_context_layerC, tf.sigmoid(tf.add(f, forget_bias_tensor))) + tf.multiply(tf.sigmoid(i), tf.tanh(j))
                new_h = tf.multiply(tf.tanh(new_c), tf.sigmoid(o))
                layer_output=new_h
                previous_context_layerC=new_c
            elif (layer<len(hids)-2):
                layer_output=af(SL)
            else:
                layer_output=SL # no activation function on final layer
                network_output_matrices.append(layer_output)
            unrolled_layers_corresponding_to_layers[layer].append(len(layer_inputs)-1)
            bias_nodes1=BL if use_shortcuts else bias_nodes
            if layer+1==context_layer:
                previous_context_layer=layer_output
            previous_layer_output=layer_output
    y=tf.stack(network_output_matrices,axis=0,name="BlockOutputMatrix")
    #print("y",y)
    y=y[delay_length:,:,:]
    #print("y",y)
    return [y, layer_sums, layer_inputs, unrolled_layers_corresponding_to_layers]

[train_inputs,train_outputs]=calculateInputOutputSequences(seqLength, batchSize, delay_length, useXor, 1, bitAdder)
[test_inputs,test_outputs]=calculateInputOutputSequences(seqLength, batchSize, delay_length, useXor, 0, bitAdder)

if use_cross_entropy:
    train_outputs=train_outputs.reshape((train_outputs.shape[0],-1))
    test_outputs=test_outputs.reshape((test_outputs.shape[0],-1))

input_realisation_network=tf.placeholder(tf.float32, [seqLength,realisation_batch_size, num_inputs_per_loop],name="input_realisation_network")
input_error_calculation_network=tf.placeholder(tf.float32, [seqLength,mini_batch_size, num_inputs_per_loop],name="input_error_calculation_network")
targets=tf.placeholder(tf.int64, [seqLength-delay_length,None],name="data_labels") if use_cross_entropy else tf.placeholder(tf.float32, [seqLength-delay_length,None, num_outputs],name="data_labels")
feed_dict_full_dataset={input_realisation_network: train_inputs[:,:realisation_batch_size,:], input_error_calculation_network: train_inputs[:,:mini_batch_size,:], targets: train_outputs[:,:mini_batch_size]}
feed_dict_full_testset={input_realisation_network: train_inputs[:,:realisation_batch_size,:], input_error_calculation_network: test_inputs[:,:mini_batch_size,:], targets: test_outputs[:,:mini_batch_size]}

batch_size=batchSize


with tf.name_scope("realisation_network"):
    # this network defines how the target matrices get converted into weight matrices
    target_matrices=build_target_matrices(input_realisation_network,hids,use_lstm,realisation_batch_size, realisation_sequence_length)
    realisation_network_weight_matrices=calculate_weight_matrices_from_targets(target_matrices, realisation_mode, input_realisation_network, hids, use_lstm, realisation_batch_size, realisation_sequence_length)
    [realisation_network_output, realisation_network_layer_sums, realisation_network_layer_inputs, _]=build_rnn(input_realisation_network, hids, realisation_network_weight_matrices, None, realisation_batch_size, realisation_sequence_length)

with tf.name_scope("error_network"):
    # this network uses the calculated weights from the above network to run a full 
    # feed-forward calculation and computation of the error function
    if use_target_space:
        ecn_weight_matrices=realisation_network_weight_matrices
    else:
        ecn_weight_matrices=build_weight_matrices(hids, use_lstm)
    [error_calculation_network_output, _, _, _]=build_rnn(input_error_calculation_network, hids, ecn_weight_matrices, None, mini_batch_size, seqLength)
    y=error_calculation_network_output



with tf.name_scope("initialise_targets"):
    initialise_targets_from_realisation_network=[tf.assign(target_matrices[i], realisation_network_layer_sums[i]) for i in range(len(target_matrices))]

with tf.name_scope("loss_function_calculation"):
    if use_cross_entropy:
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets,[-1]), logits=tf.reshape(y,[-1,2])))
        y_integer=tf.argmax(y, axis=2)
    else:
        deltas=tf.subtract(y, targets)
        squared_deltas=tf.square(deltas)
        loss=tf.reduce_sum(squared_deltas)
        y_integer=tf.cast((y>0.5), tf.float32)

    y_matches=tf.equal(y_integer, targets)
    accuracy=tf.reduce_mean(tf.cast(y_matches,tf.float32))

if use_adam:
    optimizer = tf.train.AdamOptimizer(learning_rate)
else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update = optimizer.minimize(loss, var_list=target_matrices if use_target_space else ecn_weight_matrices)
sess.run(tf.global_variables_initializer())

# it turns out to be beneficial to run the following line before starting training
if do_initial_targets_projection:
    sess.run(initialise_targets_from_realisation_network,feed_dict=feed_dict_full_dataset)
            


cumul_cpu_time=0
start_time = time.time()
print("targets,use_adam,learning_rate,pseudoinverse_regularisation_constant,iter,epoch,train_,train_loss,train_accuracy,cumul_cpu_time,af,realisation_mode,test_,test_loss,test_accuracy,xor,delay_length,sequence_length,batch_size,use_lstm,rbs,mbs,rsl,adder,targets_uses_initial_projection,targets_initialiser")

dict_recent_inputs={}
for i in range(max_its):
    batch_inputs, batch_targets=next_batch(mini_batch_size, train_inputs, train_outputs)
    feed_dict_mini_batch={input_realisation_network: train_inputs[:,:realisation_batch_size,:], input_error_calculation_network: batch_inputs, targets: batch_targets}
    if (i%100)==0: 
        cumul_cpu_time+=time.time()-start_time
        lossv,accuracyv=sess.run([loss, accuracy],feed_dict=feed_dict_full_dataset)
        tlossv,taccuracyv=sess.run([loss, accuracy],feed_dict=feed_dict_full_testset)
        #print("iteration ",i," loss",lossv, "acc", accuracyv)
        epoch=i*mini_batch_size/len(train_inputs)
        print(use_target_space,use_adam,learning_rate,pseudoinverse_regularisation_constant,i,round(epoch,4),"tr",lossv,accuracyv,cumul_cpu_time,afs,str(realisation_mode)[-3:],"test",tlossv,taccuracyv,useXor,delay_length,seqLength,batchSize,use_lstm,realisation_batch_size, mini_batch_size,realisation_sequence_length,bitAdder, do_initial_targets_projection,targets_initialiser,sep=",")
        if taccuracyv>0.99 and i in [1900,2900,4900,7900,9900,12900,16900,19900,22900]:
            break
        start_time = time.time()
        # it may or may not be beneficial to run the following line every few iterations
        #     sess.run([initialise_targets_from_current_sums],feed_dict=feed_dict_full_dataset)
    sess.run(update, feed_dict=feed_dict_mini_batch)

sess.close()
