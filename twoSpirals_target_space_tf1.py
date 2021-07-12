# This code is to accompany paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L.Citi, arXiv:2006.01578 
# This example code uses TF1 code, without using keras layers, to replicate the two-spirals experiment from the paper.  
# Please cite the above paper if this code or future variants of it are used in future academic work.
#
# Example, to run code from command line, use either:
# python3 twoSpirals_target_space_tf1.py --targets --max_its 4000 --lr 10.0 --graphical  --prc 0.001
# python3 twoSpirals_target_space_tf1.py --targets --max_its 1000 --adam --graphical 

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import time
import math
import ast
import tensorflow as tf
if float(tf.__version__[0])>=2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--targets', action='store_true', help='Use target space')
parser.add_argument('--weights', action='store_true', help='Use weight space')
parser.add_argument('--adam', action='store_true',  default=False, help='Use Adam')
parser.add_argument('--mbs', type=int, default=194, help='mini batch size (default 194)')
parser.add_argument('--lr', type=float, help='learning_rate')
parser.add_argument('--max_its', type=int, default=4000, help='max_its')
parser.add_argument('--af', type=str, default="tanh", help='activation function (tanh/relu)')
parser.add_argument('--hids', type=int, default="5", help='num hidden nodes per layer')
parser.add_argument('--ocu', action='store_true', default=False, help='use OCU realisation mode')
parser.add_argument('--bn_layers', type=str, default="[]", help='layer numbers on which to apply batch normalisation')
parser.add_argument('--avoid_projection', action='store_true', help='avoid target initial projection')
parser.add_argument('--graphical', action='store_true')
parser.add_argument('--screenshot', action='store_true')
parser.add_argument('--prc', type=float, default=0.001,  help='pseudoinverse amount of L2 regularisation')
parser.add_argument('--target_initialiser', type=float, default=1,  help='magnitude of initialised target matrices')

args = parser.parse_args()


use_target_space=args.targets or not(args.weights)
use_adam=args.adam
learning_rate=args.lr if args.lr!=None else (0.01 if use_adam else 0.1)
max_its=args.max_its
mini_batch_size=args.mbs
afs=args.af
num_hids_per_layer=args.hids
do_initial_targets_projection=not args.avoid_projection
batch_normalisation_layers=ast.literal_eval(args.bn_layers)

use_cross_entropy=True
use_shortcuts=True
hids=[2,num_hids_per_layer,num_hids_per_layer,num_hids_per_layer,2] if use_shortcuts else [2,12,12,12,2]
graphical=args.graphical


pseudoinverse_regularisation_constant=args.prc
af=tf.nn.tanh if afs=="tanh" else (tf.nn.relu if afs=="relu" else "")
weight_initialiser="GlorotNormal6" if afs=="tanh" else "HeNormal"
target_initialiser=args.target_initialiser

sess = tf.Session() 

def computeLayerWeightMatrix(target_matrix, input_matrix, regularization, name):
    with tf.name_scope("pseudoinverse_calculation"):
        weight_matrix=tf.linalg.lstsq( input_matrix,target_matrix,l2_regularizer=regularization,fast=True)
    return weight_matrix

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    return data_shuffle, labels_shuffle

def initialiser_standard_deviation(n_in, n_out):
    if weight_initialiser=="GlorotNormal6":
        return math.sqrt(6.0/(n_in+n_out))
    else:
        return math.sqrt(2.0/n_in)
        

def build_weights_matrices(hids, use_shortcuts):
    weight_matrices=[]    
    num_bias_inputs=1
    num_previous_inputs=0
    for l in range(len(hids)-1):
        num_inputs=num_bias_inputs+num_previous_inputs+hids[l]
        num_outputs=hids[l+1]
        w=tf.Variable(tf.truncated_normal([num_inputs,num_outputs], stddev=initialiser_standard_deviation(num_inputs, num_outputs),seed=None),name="WeightMatrixLayer"+str(l))
        weight_matrices.append(w)
        if use_shortcuts:
            num_previous_inputs+=hids[l]
    return weight_matrices


def build_target_matrices(batch_size,hids,use_shortcuts):
    target_matrices=[]
    for layer in range(len(hids)-1):
        TL=tf.Variable(tf.truncated_normal([batch_size,hids[layer+1]], stddev=target_initialiser, name="TargetsMatrix"+str(layer))) 
        target_matrices.append(TL)
    return target_matrices

def calculate_weight_matrices_from_targets(input_matrix, hids, use_shortcuts, target_matrices):
    calculated_weight_matrices=[]    
    layer_sums=[]
    x=input_matrix
    
    bias_nodes=tf.ones_like(x[:,0:1])
    previous_input_matrix=bias_nodes
    batch_size=x.get_shape()[0].value
    for l in range(len(hids)-1):
        with tf.name_scope("targetsRealisationLayer"+str(l)):
            if use_shortcuts:
                x=tf.concat([previous_input_matrix,x],axis=1,name="InputToLayer"+str(l))
                previous_input_matrix=x
            else:    
                x=tf.concat([bias_nodes,x],axis=1,name="InputToLayer"+str(l))

            w=computeLayerWeightMatrix(target_matrices[l],x,pseudoinverse_regularisation_constant,name="CalculatedWeightMatrixLayer"+str(l))
            calculated_weight_matrices.append(w)

            if args.ocu:
                x=target_matrices[l]
            else:
                x=tf.matmul(x,w,name="SumsInLayer"+str(l))
            layer_sums.append(x)
            if l<len(hids)-2:
                x=af(x,name="OutputFromLayer"+str(l))
    return [calculated_weight_matrices,layer_sums]

def build_ffnn(input_matrix, weight_matrices, use_shortcuts, batch_normalisation_layers, is_train):
    x=input_matrix
    layer_sums=[]
    bias_nodes=tf.ones_like(x[:,0:1])
    previous_input_matrix=bias_nodes
    for l in range(len(hids)-1):
        with tf.name_scope("layer"+str(l)):
            if use_shortcuts:
                x=tf.concat([previous_input_matrix,x],axis=1,name="InputToLayer"+str(l))
                previous_input_matrix=x
            else:    
                x=tf.concat([bias_nodes,x],axis=1,name="InputToLayer"+str(l))
            if l in batch_normalisation_layers:
                x=tf.layers.batch_normalization(x, training=is_train)
            w=weight_matrices[l]
            x=tf.matmul(x,w,name="SumsInLayer"+str(l))
            layer_sums.append(x)
            if l<len(hids)-2:
                x=af(x,name="OutputFromLayer"+str(l))
    return [x, layer_sums]

train_inputs=pd.read_csv('datasets/twoSpirals.csv',usecols = [0,1],skiprows = None,header=None).values
train_outputs = pd.read_csv('datasets/twoSpirals.csv',usecols = [2],skiprows = None ,header=None).values.reshape([-1])
test_inputs=pd.read_csv('datasets/twoSpiralsTestSet.csv',usecols = [0,1],skiprows = None,header=None).values
test_outputs = pd.read_csv('datasets/twoSpiralsTestSet.csv',usecols = [2],skiprows = None ,header=None).values.reshape([-1])
batch_size=len(train_inputs)

input_realisation_network=tf.placeholder(tf.float32, [len(train_inputs), hids[0]],name="input_realisation_network")
input_error_calculation_network=tf.placeholder(tf.float32, [None, hids[0]],name="input_error_calculation_network")
target_labels=tf.placeholder(tf.int64, [None], name="data_labels")
is_train = tf.placeholder(tf.bool, name="is_train")

feed_dict_full_dataset={input_realisation_network: train_inputs, input_error_calculation_network: train_inputs, target_labels: train_outputs, is_train:False}
feed_dict_full_testset={input_realisation_network: train_inputs, input_error_calculation_network: test_inputs, target_labels: test_outputs, is_train:False}

if use_target_space:
    with tf.name_scope("realisation_network"):
        # this network defines how the target matrices get converted into weight matrices
        target_matrices = build_target_matrices(batch_size, hids, use_shortcuts)
        [weight_matrices, realisation_network_layer_sums] = calculate_weight_matrices_from_targets(input_realisation_network, hids, use_shortcuts, target_matrices)
else:
    weight_matrices=build_weights_matrices(hids, use_shortcuts)

with tf.name_scope("error_network"):
    # this network uses the calculated weights from the above network to run a full 
    # feed-forward calculation and computation of the error function
    [error_calculation_network_output, _]=build_ffnn(input_error_calculation_network, weight_matrices, use_shortcuts, batch_normalisation_layers, is_train)
    y=error_calculation_network_output



# for plotting the image, we only want one output node, so reduce the cross-entropy two-outputs down to one as follows:
y_single_output=tf.nn.softmax(y,axis=1)[:,0]

with tf.name_scope("loss_function_calculation"):
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_labels, logits=y))
    y_integer=tf.argmax(y, axis=1)
    y_matches=tf.equal(y_integer, target_labels)
    accuracy=tf.reduce_mean(tf.cast(y_matches,tf.float32))

if use_adam:
    optimizer = tf.train.AdamOptimizer(learning_rate)
else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# the following 3 lines (1st and 3rd line only) are required for in case batch normalisation is used.
if len(batch_normalisation_layers)>0:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update = optimizer.minimize(loss)    
    update = tf.group([update, update_ops])
else:
    update = optimizer.minimize(loss)    



sess.run(tf.global_variables_initializer())

if use_target_space and do_initial_targets_projection:
    with tf.name_scope("initialise_targets"):
        initialise_targets_from_realisation_network=[tf.assign(target_matrices[i], realisation_network_layer_sums[i]) for i in range(len(target_matrices))]
    # it turns out to be beneficial to run the following line before starting training
    sess.run(initialise_targets_from_realisation_network,feed_dict=feed_dict_full_dataset)

if graphical:
    graphics_resolution=400
    axis_size=6.5
    image_input_matrix=np.array([[(y/(graphics_resolution))*axis_size*-2+axis_size,(x/(graphics_resolution))*axis_size*2-axis_size] for x in range(graphics_resolution+1) for y in range(graphics_resolution+1)],np.float32)
    plt.ion()
    fig=plt.figure("Network output")
    fig.suptitle(("Target Space" if use_target_space else "Weight Space"),y=0.92)
    y_greyscale=np.zeros((graphics_resolution+1,graphics_resolution+1),np.float32)
    plt.axis([-axis_size,axis_size,-axis_size,axis_size])
    myobj=plt.imshow((y_greyscale), cmap='gray', interpolation='none',extent=[-axis_size,axis_size,-axis_size,axis_size],vmin=0, vmax=1 )
    plt.plot(train_inputs[1::2,0], train_inputs[1::2,1],color='red',marker='o', label='Traj1', ls='',ms=4)
    plt.plot(train_inputs[0::2,0], train_inputs[0::2,1],color='blue',marker='o', label='Traj1',ls='',ms=4)
    plt.plot(test_inputs[1::2,0], test_inputs[1::2,1],color='red',marker='x', label='Traj1',ls='',ms=4)
    plt.plot(test_inputs[0::2,0], test_inputs[0::2,1],color='blue',marker='x', label='Traj1',ls='',ms=4)


cumul_cpu_time=0
start_time = time.time()
print("targets,use_adam,learning_rate,mbs,pseudoinverse_regularisation_constant,iter,epoch,train_loss,train_accuracy,cumul_cpu_time,af,num_hids_per_layer, realisation_mode,test_loss,test_accuracy,initialiser,batch_normalisation,do_initial_targets_projection")
for iteration in range(max_its):
    batch_inputs, batch_targets=next_batch(mini_batch_size, train_inputs, train_outputs)
    feed_dict_mini_batch={input_realisation_network: train_inputs, input_error_calculation_network: batch_inputs, target_labels: batch_targets, is_train: True}
    if (iteration%40)==0: 
        cumul_cpu_time+=time.time()-start_time
        lossv,accuracyv=sess.run([loss, accuracy],feed_dict=feed_dict_full_dataset)
        tlossv,taccuracyv=sess.run([loss, accuracy],feed_dict=feed_dict_full_testset)
        epoch=iteration*mini_batch_size/len(train_inputs)
        print(use_target_space,use_adam,learning_rate, mini_batch_size,pseudoinverse_regularisation_constant,iteration, round(epoch,4), lossv, accuracyv, cumul_cpu_time, afs, num_hids_per_layer,("OCU" if args.ocu else "SCU"), tlossv,taccuracyv, (weight_initialiser if use_target_space==False or target_initialiser==None else target_initialiser),str(batch_normalisation_layers).replace(" ","").replace(",","-"),do_initial_targets_projection,sep=",")
        #if accuracyv==1:
        #    break
        if graphical:
            y_greyscale=sess.run(y_single_output,feed_dict={input_realisation_network:train_inputs, input_error_calculation_network: image_input_matrix, is_train:False}).reshape(graphics_resolution+1,graphics_resolution+1)
            myobj.set_data(y_greyscale)
            plt.show()
            plt.pause(0.001)
        start_time = time.time()
        # it may or may not be beneficial to run the following line every few iterations
        #     sess.run([initialise_targets_from_current_sums],feed_dict=feed_dict_full_dataset)
    sess.run(update, feed_dict=feed_dict_mini_batch)


if graphical:
    if args.screenshot:
        import datetime
        plt.savefig("trained_net_"+("t" if use_target_space else "w"+"_"+str(datetime.datetime.now())+".png"),bbox_inches="tight")
    else:
        input("Press [enter] to continue.")
sess.close()
