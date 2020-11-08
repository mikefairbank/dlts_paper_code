# This code is to accompany paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L.Citi, arXiv:2006.01578 
# This example code uses TF1 code, without using keras layers, to replicate the imdb RNN experiment from the paper.  
# Please cite the above paper if this code or future variants of it are used in future academic work.
#
# Example, to run code from command line, use either:
#python3 rnn_imdb_target_space_tf1.py  --weights  --mbs 40 --adam --lr 0.001 --top_words 500 --max_review_length 50 --rsl 60 --max_epoch 10
#python3 rnn_imdb_target_space_tf1.py  --weights  --mbs 40 --adam --lr 0.001 --top_words 5000 --max_review_length 500 --rsl 60 --max_epoch 10
#python3 rnn_imdb_target_space_tf1.py  --targets  --mbs 40 --rbs 40 --adam --lr 0.001 --top_words 5000 --max_review_length 500 --rsl 60 --max_epoch 10
#python3 rnn_imdb_target_space_tf1.py  --targets   --mbs 40 --rbs 40 --adam --lr 0.001 --top_words 500 --max_review_length 50 --rsl 60 --max_epoch 10
#python3 rnn_imdb_target_space_tf1.py  --targets  --mbs 40 --rbs 40 --adam --lr 0.001 --top_words 50 --max_review_length 5 --rsl 6 --max_epoch 1


import sys
#sys.exit(0)
import argparse
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
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
parser.add_argument('--max_epoch', type=int)
parser.add_argument('--af', type=str, default="tanh", help='activation function (tanh/relu)')
parser.add_argument('--sequence_length', type=int, default=None, help='sequence_length')
parser.add_argument('--rsl', type=int, help='realisation_sequence_length')
parser.add_argument('--ocu', action='store_true', help='OCU algorithm (less efficient)')
parser.add_argument('--lstm', action='store_true', help='lstm')
parser.add_argument('--pic', type=float, default=0.001, help='pseudoinverse_regularisation_constant')
parser.add_argument('--avoid_projection', action='store_true', help='avoid target initial projection')
parser.add_argument('--ti', type=float, default=1.0, help='targets initialiser magnitude')
parser.add_argument('--top_words', type=int, default=500)
parser.add_argument('--max_review_length', type=int, default=50)
parser.add_argument('--embedding_length', type=int, default=32)
parser.add_argument('--context_nodes', type=int, default=100)
parser.add_argument('--realisation_input_matrices_not_embedded',action='store_true')

args = parser.parse_args()
#print("realisation_input_matrices_are_already_embedded",not(args.realisation_input_matrices_not_embedded))
#sys.exit(0)

top_words = args.top_words
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)
max_review_length = args.max_review_length 
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)
X_train=X_train.transpose().reshape([max_review_length,-1])
X_test=X_test.transpose().reshape([max_review_length,-1])
y_train=y_train.reshape([-1])
y_test=y_test.reshape([-1])


embedding_vector_length = args.embedding_length
use_target_space=args.targets or not(args.weights)
use_adam=args.adam
learning_rate=args.lr if args.lr!=None else ((0.01 if use_target_space else 0.001) if use_adam  else 0.1)
mini_batch_size=args.mbs if args.mbs!=None else batchSize
max_its=args.max_its
if args.max_epoch !=None:
	max_its=(X_train.shape[1]*args.max_epoch)//mini_batch_size
batchSize=X_train.shape[1]
realisation_batch_size=args.rbs if args.rbs!=None else mini_batch_size
afs=args.af
use_lstm=args.lstm

seqLength=max_review_length
if args.rsl!=None:
	realisation_sequence_length=args.rsl
else:
	realisation_sequence_length=seqLength
do_initial_targets_projection=not args.avoid_projection
use_shortcuts=False
realisation_input_matrices_are_already_embedded=not(args.realisation_input_matrices_not_embedded)


num_inputs_per_loop=embedding_vector_length
num_outputs=2
numContextNodes=args.context_nodes 
hids=[num_inputs_per_loop, numContextNodes,num_outputs]
context_layer=1
datatype_tf=tf.float32 
targets_initialiser=args.ti



pseudoinverse_regularisation_constant=args.pic
af=tf.nn.tanh if afs=="tanh" else (tf.nn.relu if afs=="relu" else "")
weight_initialiser="GlorotNormal6" if afs=="tanh" else "HeNormal"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

def computeLayerWeightMatrix(target_matrix, input_matrix, regularization, name):
	with tf.name_scope("pseudoinverse_calculation"):
		weight_matrix=tf.linalg.lstsq( input_matrix,target_matrix,l2_regularizer=regularization,fast=True)
	return weight_matrix

def next_batch(num, data, labels):
    idx = np.arange(0 , data.shape[1])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[:,idx]
    labels_shuffle = labels[idx]
    return data_shuffle, labels_shuffle

def initialiser_standard_deviation(n_in, n_out):
	return math.sqrt((6.0 if weight_initialiser=="GlorotNormal6" else 2.0)/(n_in+(n_out if weight_initialiser=="GlorotNormal6" else 0)))



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
	embedding_weights_matrix=tf.Variable(tf.truncated_normal([top_words,embedding_vector_length], stddev=0.1, dtype=datatype_tf, name="EmbeddingMatrix"))
	return [calculated_weight_matrices, embedding_weights_matrix]

def build_target_matrices(input_matrix,hids,use_lstm,mini_batch_size, seqLength, input_matrices_are_already_embedded):
	layer_target_matrices=[]
	# this was an incredibly lazy way to find the dimensions of the target matrices....
	[temp_weight_matrices, embedding_matrix]=build_weight_matrices(hids, use_lstm)
	[_, layer_sums_temp, _, unrolled_layers_corresponding_to_layers]=build_rnn(input_matrix, hids, temp_weight_matrices, embedding_matrix, None, mini_batch_size, seqLength, input_matrices_are_already_embedded)
	for layer in range(len(layer_sums_temp)):
		SL=layer_sums_temp[layer]
		TL=tf.Variable(tf.truncated_normal(SL.get_shape(), stddev=targets_initialiser, dtype=datatype_tf, name="TargetsMatrix"+str(layer))) 
		layer_target_matrices.append(TL)
	#print("Built target matrices",layer_target_matrices)
	return [layer_target_matrices,embedding_matrix]

def calculate_weight_matrices_from_targets(layer_target_matrices, embedding_weights_matrix, input_matrix, hids, use_lstm, mini_batch_size, seqLength, input_matrices_are_already_embedded):
	predicted_sums_from_targets=layer_target_matrices
	[temp_weight_matrices,_]=build_weight_matrices(hids, use_lstm)
	# push the current targets through the network, and force them to become the "sums", so that the output activations of every layer can be approximated
	# and so that the variable layer_inputs_temp can be found
	[_, layer_sums_temp, layer_inputs_temp, unrolled_layers_corresponding_to_layers_temp]=build_rnn(input_matrix, hids, temp_weight_matrices, embedding_weights_matrix, predicted_sums_from_targets, mini_batch_size, seqLength, input_matrices_are_already_embedded)

	calculated_weight_matrices=[]
	if args.ocu:
		for layer in range(len(hids)-1):
			#print("layer",layer)
			#print("layer inputs",[layer_inputs[x] for x in unrolled_layers_corresponding_to_layers[layer]])
			BL_concatenated=tf.concat([layer_inputs_temp[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
			TL_concatenated=tf.concat([layer_target_matrices[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
			WL=computeLayerWeightMatrix(TL_concatenated, BL_concatenated, pseudoinverse_regularisation_constant, "CalculateWeightMatrix"+str(layer))
			calculated_weight_matrices.append(WL)
	else:
		# SCU + OCU hybrid
		#First layer uses OCU:
		layer=0
		BL_concatenated=tf.concat([layer_inputs_temp[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
		TL_concatenated=tf.concat([layer_target_matrices[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
		WL=computeLayerWeightMatrix(TL_concatenated, BL_concatenated, pseudoinverse_regularisation_constant, "CalculateWeightMatrix"+str(layer))
		calculated_weight_matrices.append(WL)

		temp_weight_matrices[0:len(calculated_weight_matrices)]=calculated_weight_matrices
		if context_layer!=1:
			raise Exception("Not implemented yet2")
		[_, layer_sums_temp, layer_inputs_temp, unrolled_layers_corresponding_to_layers_temp]=build_rnn(input_matrix, hids, temp_weight_matrices, embedding_weights_matrix, None, mini_batch_size, seqLength, input_matrices_are_already_embedded)
		layer=1
		BL_concatenated=tf.concat([layer_inputs_temp[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
		TL_concatenated=tf.concat([layer_target_matrices[x] for x in unrolled_layers_corresponding_to_layers_temp[layer]],axis=0)
		WL=computeLayerWeightMatrix(TL_concatenated, BL_concatenated, pseudoinverse_regularisation_constant, "CalculateWeightMatrix"+str(layer))
		calculated_weight_matrices.append(WL)
		if layer!=len(hids)-2:
			raise Exception("Not implemented yet3")
	return calculated_weight_matrices


def build_rnn(input_matrix, hids, weight_matrices, embedding_matrix, given_loop_sums, mini_batch_size, seqLength, input_matrices_are_already_embedded):
	# Build recurrent neural network....
	#initialHiddenState=tf.constant(0., shape=[mini_batch_size, numContextNodes], dtype=datatype_tf) 
	initialHiddenState=tf.fill(dims=[tf.shape(input_matrix)[1],numContextNodes], value=0.0)
	if use_lstm:
		initialHiddenCState=tf.fill(dims=[tf.shape(input_matrix)[1],numContextNodes], value=0.0)
		previous_context_layerC=initialHiddenCState
	#print("initialHiddenState",initialHiddenState,"numContextNodes",numContextNodes)
	#print("input_matrix",tf.shape(input_matrix))
	#print("input_matrix",(input_matrix))
	#sys.exit(0)
	previous_context_layer=initialHiddenState # This must not be tf.Variable
	unrolled_layers_corresponding_to_layers=[[] for _ in range(len(hids)-1)]
	bias_nodes=None
	layer_sums=[]
	layer_inputs=[]
	network_output_matrices=[]
	for loop in range(seqLength):
		if input_matrices_are_already_embedded:
			layer_input_matrix=input_matrix[loop,:,:]
		else:
			layer_input_matrix=tf.gather(embedding_matrix, indices=input_matrix[loop,:],axis=0)
		if bias_nodes==None:
			bias_nodes=tf.ones_like(layer_input_matrix[:,0:1])
		#print("input_matrix[loop,:]",input_matrix[loop,:])
		#print("embedding_matrix",embedding_matrix)
		#print("layer_input_matrix", layer_input_matrix)
		previous_layer_output=tf.concat([layer_input_matrix,previous_context_layer],axis=1)
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
	y=network_output_matrices[-1]
	return [y, layer_sums, layer_inputs, unrolled_layers_corresponding_to_layers]

[train_inputs,train_outputs]=[X_train, y_train]
[test_inputs,test_outputs]=[X_test, y_test]
train_outputs=train_outputs.reshape((-1))
test_outputs=test_outputs.reshape((-1))

if realisation_input_matrices_are_already_embedded:
	input_realisation_network=tf.placeholder(tf.float32, [realisation_sequence_length,realisation_batch_size,embedding_vector_length],name="input_realisation_network")
	np_realisation_inputs=np.random.rand(realisation_sequence_length,realisation_batch_size,embedding_vector_length)*2-1
else:
	input_realisation_network=tf.placeholder(tf.int32, [realisation_sequence_length,realisation_batch_size],name="input_realisation_network")
	np_realisation_inputs=train_inputs[:,:realisation_batch_size]

ph_input_error_calculation_network=tf.placeholder(tf.int32, [seqLength,None],name="ph_input_error_calculation_network")
ph_data_labels=tf.placeholder(tf.int64, [None],name="data_labels")
feed_dict_full_dataset={input_realisation_network: np_realisation_inputs, ph_input_error_calculation_network: train_inputs[:,:mini_batch_size], ph_data_labels: train_outputs[:mini_batch_size]}
feed_dict_full_testset={input_realisation_network: np_realisation_inputs, ph_input_error_calculation_network: test_inputs[:,:mini_batch_size], ph_data_labels: test_outputs[:mini_batch_size]}

batch_size=batchSize


with tf.name_scope("realisation_network"):
	# this network defines how the target matrices get converted into weight matrices
	[target_matrices,realisation_network_embedding_matrix]=build_target_matrices(input_realisation_network,hids,use_lstm,realisation_batch_size, realisation_sequence_length,realisation_input_matrices_are_already_embedded)
	realisation_network_weight_matrices=calculate_weight_matrices_from_targets(target_matrices, realisation_network_embedding_matrix, input_realisation_network, hids, use_lstm, realisation_batch_size, realisation_sequence_length,realisation_input_matrices_are_already_embedded)
	[realisation_network_output, realisation_network_layer_sums, realisation_network_layer_inputs, _]=build_rnn(input_realisation_network, hids, realisation_network_weight_matrices, realisation_network_embedding_matrix, None, realisation_batch_size, realisation_sequence_length, realisation_input_matrices_are_already_embedded)


with tf.name_scope("error_network"):
	# this network uses the calculated weights from the above network to run a full 
	# feed-forward calculation and computation of the error function
	#def build_rnn(input_matrix, hids, weight_matrices, embedding_weights_matrix, given_loop_sums, build_target_matrices, build_weight_matrices, mini_batch_size, seqLength):
	if use_target_space:
		[ecn_weight_matrices,ecn_embedding_matrix]=[realisation_network_weight_matrices,realisation_network_embedding_matrix]
	else:
		[ecn_weight_matrices,ecn_embedding_matrix]=build_weight_matrices(hids, use_lstm)
	[error_calculation_network_output, _, _, _]=build_rnn(ph_input_error_calculation_network, hids, ecn_weight_matrices, ecn_embedding_matrix, None, mini_batch_size, seqLength, False)
	y=error_calculation_network_output



with tf.name_scope("initialise_targets"):
	initialise_targets_from_realisation_network=[tf.assign(target_matrices[i], realisation_network_layer_sums[i]) for i in range(len(target_matrices))]

with tf.name_scope("loss_function_calculation"):
	loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(ph_data_labels,[-1]), logits=tf.reshape(y,[-1,2])))
	y_integer=tf.argmax(y, axis=1)
	y_matches=tf.equal(y_integer, ph_data_labels)
	accuracy=tf.reduce_mean(tf.cast(y_matches,tf.float32))

if use_adam:
	optimizer = tf.train.AdamOptimizer(learning_rate)
else:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)

update = optimizer.minimize(loss, var_list=target_matrices+[realisation_network_embedding_matrix] if use_target_space else ecn_weight_matrices+[ecn_embedding_matrix])
sess.run(tf.global_variables_initializer())

# it turns out to be beneficial to run the following line before starting training
if do_initial_targets_projection:
	if targets_initialiser!=None:
		sess.run(initialise_targets_from_realisation_network,feed_dict=feed_dict_full_dataset)
	else:
		sess.run(initialise_targets_from_other_network_sums,feed_dict=feed_dict_full_dataset)
			

cumul_cpu_time=0
start_time = time.time()
print("targets,use_adam,learning_rate,pseudoinverse_regularisation_constant,iter,epoch,train_,lossv,accuracy,cumul_cpu_time,af, realisation_mode,test_,test_loss,test_accuracy,sequence_length,batch_size,use_lstm,rbs,mbs,rsl,targets_uses_initial_projection, targets_initialiser,top_words,max_review_length,embedding_length,realisation_input_matrices_are_already_embedded,context_nodes")

dict_recent_inputs={}
for i in range(max_its):
	batch_inputs, batch_labels=next_batch(mini_batch_size, train_inputs, train_outputs)
	#print("batch_inputs",batch_inputs.shape)
	#print("ph_input_error_calculation_network", ph_input_error_calculation_network)
	#print("mini_batch_size",mini_batch_size)
	#print("train_inputs",train_inputs.shape)
	feed_dict_mini_batch={input_realisation_network: np_realisation_inputs, ph_input_error_calculation_network: batch_inputs, ph_data_labels: batch_labels}
	if (i%50)==0: 
		cumul_cpu_time+=time.time()-start_time
		num_trials=5
		trial_batch_size=5000
		if num_trials*trial_batch_size!=test_inputs.shape[1]:
			raise Exception("error num_trials*trial_batch_size!=test_inputs.shape[1]")
		lossv=0
		accuracyv=0
		tlossv=0
		taccuracyv=0
		# We're splitting the test set into mini-batches, and then accumulating the accuracy across all batches, purely to save memory
		for trial in range(num_trials):
			# training set...
			feed_dict_temp={input_realisation_network: np_realisation_inputs, ph_input_error_calculation_network: train_inputs[:,trial*trial_batch_size:(trial+1)*trial_batch_size], ph_data_labels: train_outputs[trial*trial_batch_size:(trial+1)*trial_batch_size]}
			lossv_,accuracyv_=sess.run([loss, accuracy],feed_dict=feed_dict_temp)
			lossv+=lossv_
			accuracyv+=accuracyv_

			# test set....
			feed_dict_temp={input_realisation_network: np_realisation_inputs, ph_input_error_calculation_network: test_inputs[:,trial*trial_batch_size:(trial+1)*trial_batch_size], ph_data_labels: test_outputs[trial*trial_batch_size:(trial+1)*trial_batch_size]}
			tlossv_,taccuracyv_=sess.run([loss, accuracy],feed_dict=feed_dict_temp)
			tlossv+=tlossv_
			taccuracyv+=taccuracyv_
		lossv/=num_trials
		accuracyv/=num_trials
		tlossv/=num_trials
		taccuracyv/=num_trials

		#print("iteration ",i," loss",lossv, "acc", accuracyv)
		epoch=i*mini_batch_size/train_inputs.shape[1]
		#print("i",mini_batch_size,train_inputs.shape,train_inputs.shape[1],mini_batch_size/train_inputs.shape[1])
		#print("epoch",epoch)
		print(use_target_space,use_adam,learning_rate,pseudoinverse_regularisation_constant,i,round(epoch,4),"tr",round(lossv,4), round(accuracyv,4), cumul_cpu_time, afs, "OCU" if args.ocu else "SCU", "test",round(tlossv,4),round(taccuracyv,4), seqLength, batchSize, use_lstm, realisation_batch_size, mini_batch_size,realisation_sequence_length, do_initial_targets_projection, targets_initialiser,top_words,max_review_length, embedding_vector_length, realisation_input_matrices_are_already_embedded, numContextNodes, sep=",")
		start_time = time.time()
	sess.run(update, feed_dict=feed_dict_mini_batch)

sess.close()
