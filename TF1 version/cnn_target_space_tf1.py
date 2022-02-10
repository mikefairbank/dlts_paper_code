# This code is to accompany the paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L. Citi, https://jmlr.org/papers/v23/20-040.html 
# This example code uses TF1 code, without using keras layers, to replicate the CNN classification experiments from the paper.  
# Please cite the above paper if this code or future variants of it are used in future academic work.
#
# To run on the cifar100 dataset, use:
# python3 cnn_target_space_tf1.py --dense_dims [512,128,100] --conv_dims [[3,3,3,3,3,3],[1,2,1,2,1,2],[32,32,64,64,128,128]] --targets --adam --max_epoch 400 --mbs 100 --realisation_batch_size 100 --prc 0.1 --af lrelu --dataset cifar100 
# To test on MNIST with a tiny CNN use:
# python3 cnn_target_space_tf1.py --dense_dims [100] --conv_dims [[3],[1],[16]] --targets --adam --max_epoch 400 --mbs 100 --realisation_batch_size 100 --prc 0.1 --af lrelu --dataset cifar100 


import numpy as np
import sys
import argparse
import time
import math
import ast
import tensorflow as tf
if float(tf.__version__[0])>=2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
    
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--targets', action='store_true', help='Use target space')
parser.add_argument('--weights', action='store_true', help='Use weight space')
parser.add_argument('--adam', action='store_true',  help='Use Adam')
parser.add_argument('--mbs', type=int, default=10, help='mini batch size (default 10)')
parser.add_argument('--lr', type=float, help='learning_rate')
parser.add_argument('--max_its', type=int, help='max_its')
parser.add_argument('--max_epoch', type=int, default=0,help='max_epoch')
parser.add_argument('--keep_prob', type=str, default=None,help='keep_prob')
parser.add_argument('--double_keep_prob', type=str, default=None,help='keep_prob_double')
parser.add_argument('--realisation_keep_prob', type=str, default=None,help='keep_prob_double')
parser.add_argument('--af', type=str, default="relu", help='activation function (tanh/relu/lrelu)')
parser.add_argument('--realisation_batch_size', type=int, default=100, help='realisation_batch_size')
parser.add_argument('--dense_dims', type=str, help='dims for dense NN e.g. [784,100,50,10,10]')
parser.add_argument('--conv_dims', type=str, help='dims for conv NN e.g. [[5,5],[2,2],[32,64]]')
parser.add_argument('--prc', type=float, default=0.1, help="pseudoinverse_regularisation_constant")
parser.add_argument('--ti', type=float, default=0.1, help="target initialiser")
parser.add_argument('--avoid_projection', action='store_true', help='avoid target initial projection')
parser.add_argument('--dataset', type=str, default="mnist", help='dataset cifar100/mnist/cifar10/fashion')
parser.add_argument('--bn_layers', type=str, default="[]", help='layer numbers on which to apply batch normalisation')

args = parser.parse_args()


dataset_name=""
if args.dataset=="cifar10":
    dataset_name="cifar10"
    cifar10 = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    input_image_side_length=32
    input_image_channels=3
    # CIFAR10 images are 32*32*3. 
elif args.dataset=="cifar100":
    dataset_name="cifar100"
    cifar100 = tf.keras.datasets.cifar100
    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode="fine" )
    input_image_side_length=32
    input_image_channels=3
    # CIFAR10 images are 32*32*3. 
elif args.dataset=="fashion":
    dataset_name="fashion"
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    input_image_side_length=28
    input_image_channels=1
    train_images=train_images.reshape(list(train_images.shape)+[1])
    test_images=test_images.reshape(list(test_images.shape)+[1])
elif args.dataset=="mnist":
    dataset_name="mnist"
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
    input_image_side_length=28
    input_image_channels=1
    train_images=train_images.reshape(list(train_images.shape)+[1])
    test_images=test_images.reshape(list(test_images.shape)+[1])
    # MNIST images are 28*28*1.  
else:
    print("unknown dataset")
    sys.exit(0)



# Rescale greyscale from 8 bit to floating point (by dividing by 255)
test_images=(test_images/255.0).astype(np.float32) # 10000 test patterns, shape 10000*28*28  
train_images=(train_images/255.0).astype(np.float32) # 60000 train patterns, shape 60000*28*28
train_labels=train_labels.reshape(-1)
test_labels=test_labels.reshape(-1)

# createa a validation_set by taking final 5000 images from training set
test_images1,test_labels1=train_images[-5000:],train_labels[-5000:]
#test_images1,test_labels1=train_images[-100:],train_labels[-100:]
train_images,train_labels=train_images[:-5000],train_labels[:-5000]
#test set:
test_images2,test_labels2=test_images,test_labels
#test_images2,test_labels2=test_images[:100],test_labels[:100]
num_classification_categories=train_labels.max()+1

split_test_set_into_chunks=5


use_target_space=args.targets or not(args.weights)
use_adam=args.adam
afs=args.af
af=tf.nn.tanh if afs=="tanh" else (tf.nn.relu if afs=="relu" else (tf.nn.leaky_relu if afs=="lrelu" else ""))
#if afs=="tanh" or args.dense_dims=="[100-10]":
#    sys.exit(0)
learning_rate=args.lr if args.lr!=None else (0.01 if (use_adam and use_target_space) else (0.001 if use_adam else 0.1))
#use_shortcuts=True
pseudoinverse_regularisation_constant=args.prc # changed from 0.1 22-Oct-2019.   
mini_batch_size = 10 if args.mbs==None else args.mbs
training_iters = (1000000 *(1 if use_target_space else 10)//mini_batch_size)//(10 if use_adam else 1) if args.max_its==None else args.max_its
if args.max_epoch>0:
    training_iters=len(train_images)*args.max_epoch//mini_batch_size
realisation_batch_size=args.realisation_batch_size
training_keep_prob=ast.literal_eval(args.keep_prob) if args.keep_prob!=None else None
training_keep_prob_double=ast.literal_eval(args.double_keep_prob) if args.double_keep_prob!=None else None
training_keep_prob_realisation=ast.literal_eval(args.realisation_keep_prob) if args.realisation_keep_prob!=None else None
use_double_keep_prob=training_keep_prob_double !=None
use_realisation_keep_prob=training_keep_prob_realisation !=None
if training_keep_prob_double!=None:
    training_keep_prob=training_keep_prob_double
if training_keep_prob_realisation!=None:
    training_keep_prob=training_keep_prob_realisation
do_initial_targets_projection=not args.avoid_projection
weight_initialiser="GlorotNormal" if afs=="tanh" else "HeNormal"
target_initialiser=args.ti
batch_normalisation_layers=ast.literal_eval(args.bn_layers)
if len(batch_normalisation_layers)>0 and use_target_space:
    print("Illegal argument: Batch Norm with Target Space")
    sys.exit(0)
import tensorflow.linalg as lstsq_alg

[conv_kernel_side_length_per_layer, max_pool_side_length_per_layer, channels_per_layer]=[[5,5],[2,2],[32,64]]
    

fully_connected_hidden_nodes_per_layer=[1024,num_classification_categories]
if args.dense_dims!=None:
    [conv_kernel_side_length_per_layer, max_pool_side_length_per_layer, channels_per_layer]=[[],[],[]]
    fully_connected_hidden_nodes_per_layer=ast.literal_eval(args.dense_dims)
    if fully_connected_hidden_nodes_per_layer[-1]!=num_classification_categories:
        raise Exception("illegal dims for final dense layer - need "+str(num_classification_categories)+" outputs")

if args.conv_dims!=None:
    [conv_kernel_side_length_per_layer, max_pool_side_length_per_layer, channels_per_layer]=ast.literal_eval(args.conv_dims)
total_num_network_layers=len(conv_kernel_side_length_per_layer)+len(fully_connected_hidden_nodes_per_layer)


if training_keep_prob == None:
    dropout_keep_probs_training=None
    dropout_keep_probs_testing=None
else:
    if type(training_keep_prob) is list:
        dropout_keep_probs_training=(training_keep_prob+[1.0]*total_num_network_layers)[:total_num_network_layers]
    else:
        dropout_keep_probs_training=([training_keep_prob]*(total_num_network_layers-1))+[1.0]
    #print("dropout_keep_probs_training",dropout_keep_probs_training)
    dropout_keep_probs_testing=[1]*total_num_network_layers

sess = tf.Session() 

def computeLayerWeightMatrix(target_matrix, input_matrix, l2_regularizer, name):
    weight_matrix=lstsq_alg.lstsq( input_matrix,target_matrix,l2_regularizer) 
    return weight_matrix

def computeLayerWeightMatrixCNN(target_matrix, input_image, kernel_side_length, l2_regularizer, name):
    num_channels=input_image.get_shape()[3].value
    input_patches=tf.image.extract_image_patches(images=input_image, ksizes=[1,kernel_side_length,kernel_side_length,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME")
    flattened_input=tf.reshape(input_patches,[-1,kernel_side_length*kernel_side_length*num_channels])
    flattened_input=tf.concat([tf.ones_like(flattened_input[:,0:1]),flattened_input],axis=1)
    return lstsq_alg.lstsq(flattened_input, target_matrix, l2_regularizer)

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    return data_shuffle, labels_shuffle

def initialiser_standard_deviation(n_in, n_out):
    return math.sqrt(2.0/(n_in+(n_out if weight_initialiser=="GlorotNormal" else 0)))
#def initialiser_standard_deviation(n_in, n_out):
#    return math.sqrt((6.0 if weight_initialiser=="GlorotNormal6" else 2.0)/(n_in+(n_out if weight_initialiser=="GlorotNormal6" else 0)))


def generate_convolutional_kernel_weights(conv_kernel_side_length, input_channels, output_channels,name):
    n_in=1+conv_kernel_side_length*conv_kernel_side_length*input_channels
    n_out=output_channels
    weights=tf.Variable(tf.random_normal([n_in, n_out], stddev=initialiser_standard_deviation(n_in, n_out)),name=name)
    return weights
    
def generate_fully_connected_weights(num_inputs, num_outputs, name):
    weights=tf.Variable(tf.random_normal([1+num_inputs, num_outputs], stddev=initialiser_standard_deviation(1+num_inputs, num_outputs)),name=name)
    return weights

def convert_list_to_csv(list1):
    list1=[str(x) for x in list1]
    if len(list1)==0:
        return ""
    return ",".join(list1)+","

    
def build_weight_matrices():
    weight_matrices=[]
    num_channels=input_image_channels
    image_side_length=input_image_side_length
    for l in range(len(conv_kernel_side_length_per_layer)):
        w_reshaped=generate_convolutional_kernel_weights(conv_kernel_side_length_per_layer[l], num_channels, channels_per_layer[l], name="ConvWeightMatrixLayer"+str(l))
        weight_matrices.append(w_reshaped)
        num_channels=channels_per_layer[l]
        image_side_length=math.ceil(image_side_length/max_pool_side_length_per_layer[l])
    num_nodes=num_channels*image_side_length*image_side_length
    for l in range(len(fully_connected_hidden_nodes_per_layer)):
        w=generate_fully_connected_weights(num_nodes, fully_connected_hidden_nodes_per_layer[l], name="DenseWeightMatrixLayer"+str(l))
        num_nodes=fully_connected_hidden_nodes_per_layer[l]
        weight_matrices.append(w)
    return weight_matrices

def build_target_matrices():
    target_matrices=[]
    num_channels=input_image_channels
    image_side_length=input_image_side_length
    for l in range(len(conv_kernel_side_length_per_layer)):
        # because convolutional padding is "same", it means the number of patches = number of pixels in original image, i.e.
        num_patches=image_side_length**2
        target_matrices.append(tf.Variable(tf.truncated_normal([realisation_batch_size*num_patches,channels_per_layer[l]], stddev=target_initialiser),name="TargetMatrixConvayer"+str(l)))
        num_channels=channels_per_layer[l]
        image_side_length=math.ceil(image_side_length/max_pool_side_length_per_layer[l])
    num_nodes=num_channels*image_side_length*image_side_length
    for l in range(len(fully_connected_hidden_nodes_per_layer)):
        target_matrices.append(tf.Variable(tf.truncated_normal([realisation_batch_size, fully_connected_hidden_nodes_per_layer[l]], stddev=target_initialiser),name="TargetMatrixDenseLayer"+str(l+len(conv_kernel_side_length_per_layer))))
    return target_matrices
    
def convert_target_matrices_to_weights(x, layer_target_matrices, keep_probs):
    x=tf.reshape(x,[-1, input_image_side_length,input_image_side_length,input_image_channels])
    num_channels=input_image_channels
    image_side_length=input_image_side_length
    layer_outputs=[]
    calculated_weight_matrices=[]
    with tf.name_scope("convert_target_matrices_to_weights"):
        with tf.name_scope("convolutional_layers"):
            for l in range(len(conv_kernel_side_length_per_layer)):
                with tf.name_scope("layer"+str(l)):
                    k=conv_kernel_side_length_per_layer[l]
                    target_matrix=layer_target_matrices[l]
                    w_reshaped=computeLayerWeightMatrixCNN(target_matrix, x, k, pseudoinverse_regularisation_constant, name="CalculatedWeightMatrixLayer"+str(l))
                    b=w_reshaped[0,:]
                    W=tf.reshape(w_reshaped[1:,:],[conv_kernel_side_length_per_layer[l],conv_kernel_side_length_per_layer[l], num_channels, channels_per_layer[l]])
                    strides=1
                    test_output_matrix = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME',name="ConvLayer"+str(l))
                    test_output_matrix = tf.nn.bias_add(test_output_matrix, b[:])
                    test_output_matrix=tf.reshape(test_output_matrix,[-1,channels_per_layer[l]])
                    calculated_weight_matrices.append(w_reshaped)
                    layer_outputs.append(test_output_matrix)
                    x=tf.reshape(test_output_matrix,[-1, image_side_length, image_side_length, channels_per_layer[l]])
                    x = af(x)
                    k=max_pool_side_length_per_layer[l]
                    if k!=None and k>1:
                        x=tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME',name="MaxPoolLayer"+str(l))
                        image_side_length=x.get_shape()[1].value
                    if keep_probs!=None:
                        x = tf.nn.dropout(x, keep_probs[l])
                    num_channels=channels_per_layer[l]
            with tf.name_scope("flatten"):
                x=tf.reshape(x,[-1, x.get_shape()[1].value*x.get_shape()[2].value*x.get_shape()[3].value])
        with tf.name_scope("dense_layers"):
            num_nodes=num_channels*image_side_length*image_side_length
            for l in range(len(fully_connected_hidden_nodes_per_layer)):
                with tf.name_scope("layer"+str(l)):
                    x=tf.concat([tf.ones_like(x[:,0:1]),x],axis=1,name="InputToLayer"+str(l))
                    w=computeLayerWeightMatrix(layer_target_matrices[l+len(conv_kernel_side_length_per_layer)],x,pseudoinverse_regularisation_constant, name="CalculatedWeightMatrixLayer"+str(l))
                    calculated_weight_matrices.append(w)
                    x=tf.matmul(x,w,name="SumsInLayer"+str(l))
                    layer_outputs.append(x)
                    if l<len(fully_connected_hidden_nodes_per_layer)-1:
                        x=af(x,name="OutputFromLayer"+str(l))
                    if keep_probs!=None:
                        x = tf.nn.dropout(x, keep_probs[l+len(conv_kernel_side_length_per_layer)])
                    num_nodes=fully_connected_hidden_nodes_per_layer[l]
                    
    return [calculated_weight_matrices, layer_outputs]


def build_cnn(x, weight_matrices, keep_probs, is_testing, batch_normalisation_layers, is_train):
    x=tf.reshape(x,[-1, input_image_side_length,input_image_side_length,input_image_channels])
    num_channels=input_image_channels
    image_side_length=input_image_side_length
    layer_outputs=[]

    with tf.name_scope("convolutional_layers"):
        for l in range(len(conv_kernel_side_length_per_layer)):
            with tf.name_scope("layer"+str(l)):
                w_reshaped=weight_matrices[l]
                b=w_reshaped[0,:]
                W=tf.reshape(w_reshaped[1:,:],[conv_kernel_side_length_per_layer[l],conv_kernel_side_length_per_layer[l], num_channels, channels_per_layer[l]])
                strides=1
                x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME',name="ConvLayer"+str(l))
                x = tf.nn.bias_add(x, b)
                x = af(x)
                if l in batch_normalisation_layers:
                    x=tf.layers.batch_normalization(x, training=is_train)
                k=max_pool_side_length_per_layer[l]
                if k!=None and k>1:
                    x=tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME',name="MaxPoolLayer"+str(l))
                    image_side_length=x.get_shape()[2].value
                if keep_probs!=None:
                    x = tf.nn.dropout(x, keep_probs[l])
                num_channels=channels_per_layer[l]
        with tf.name_scope("flatten"):
            x=tf.reshape(x,[-1, x.get_shape()[1].value*x.get_shape()[2].value*x.get_shape()[3].value])
    with tf.name_scope("dense_layers"):
        num_nodes=num_channels*image_side_length*image_side_length
        for l in range(len(fully_connected_hidden_nodes_per_layer)):
            with tf.name_scope("layer"+str(l)):
                x=tf.concat([tf.ones_like(x[:,0:1]),x],axis=1,name="InputToLayer"+str(l))
                w=weight_matrices[l+len(conv_kernel_side_length_per_layer)]
                x=tf.matmul(x,w,name="SumsInLayer"+str(l))
                layer_outputs.append(x)
                if l<len(fully_connected_hidden_nodes_per_layer)-1:
                    x=af(x,name="OutputFromLayer"+str(l))
                if l+len(conv_kernel_side_length_per_layer) in batch_normalisation_layers:
                    x=tf.layers.batch_normalization(x, training=is_train)
                if keep_probs!=None:
                    x = tf.nn.dropout(x, keep_probs[l+len(conv_kernel_side_length_per_layer)])
                num_nodes=fully_connected_hidden_nodes_per_layer[l]
    return [x, layer_outputs]



input_realisation_network=tf.placeholder(tf.float32, [realisation_batch_size, input_image_side_length, input_image_side_length,input_image_channels],name="input_realisation_network")
input_error_calculation_network=tf.placeholder(tf.float32, [None, input_image_side_length, input_image_side_length,input_image_channels],name="input_error_calculation_network")
target_labels = tf.placeholder(tf.int64, [None],name="data_labels")
ph_use_identical_patterns_across_ensemble = tf.placeholder(tf.bool, [], name="ph_testing")
keep_prob_ph = tf.placeholder(tf.float32, [len(training_keep_prob)] if training_keep_prob!=None else [],name="keep_probs")  
is_train = tf.placeholder(tf.bool, name="is_train")


with tf.name_scope("realisation_network"):
    # this network defines how the target matrices get converted into weight matrices
    layer_target_matrices=build_target_matrices()
    [calculated_weight_matrices_realisation_network, layer_outputs_realisation_network]=convert_target_matrices_to_weights(input_realisation_network, layer_target_matrices, (keep_prob_ph if training_keep_prob!=None else None) if (use_double_keep_prob or use_realisation_keep_prob) else dropout_keep_probs_testing)
    with tf.name_scope("initialise_targets"):
        if not args.avoid_projection:
            initialise_targets_from_current_sums=[tf.assign(layer_target_matrices[i], layer_outputs_realisation_network[i]) for i in range(len(layer_target_matrices))]

# this network uses the calculated weights from the above network to run a full 
# feed-forward calculation and computation of the error function
with tf.name_scope("error_network"):
    if use_target_space:
        error_network_weight_matrices=calculated_weight_matrices_realisation_network
    else:
        error_network_weight_matrices=build_weight_matrices()
    [ecn_output,_]=build_cnn(input_error_calculation_network, error_network_weight_matrices, keep_prob_ph if (training_keep_prob!=None and not use_realisation_keep_prob) else None, ph_use_identical_patterns_across_ensemble, batch_normalisation_layers, is_train)
    y=ecn_output


with tf.name_scope("loss_function_calculation"):
    loss_mean=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_labels[:], logits=y[:,:]))
    y_integer=tf.argmax(y, axis=1)
    y_matches=tf.equal(y_integer, target_labels)
    accuracy=tf.reduce_mean(tf.cast(y_matches,tf.float32))


if use_adam:
    optimizer = tf.train.AdamOptimizer(learning_rate)
else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)


# the following 3 lines (1st and 3rd line only) are required for in case batch normalisation is used.
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update = optimizer.minimize(loss_mean)
if batch_normalisation_layers!=[]:
    update = tf.group([update, update_ops])

sess.run(tf.global_variables_initializer())

# it turns out to be beneficial to run the following line before starting training
if do_initial_targets_projection:
    sess.run(initialise_targets_from_current_sums,feed_dict={input_realisation_network: train_images[:realisation_batch_size], input_error_calculation_network: train_images[:realisation_batch_size], target_labels: train_labels[:realisation_batch_size], keep_prob_ph: dropout_keep_probs_testing, is_train: False})


cumul_cpu_time=0
start_time = time.time()
print("dataset_name,target_space,use_adam,learning_rate,realisation_batch_size,pseudoinverse_regularisation_constant,mbs,iter,epoch,tr_,loss_tr,acc_tr,valid_,loss_valid,acc_valid,"+
    "test_,loss_test,acc_test,cumul_cpu_time,keep_probs,initaliser,af,dense_nodes,conv_nodes,batch_normalisation_layers,do_initial_targets_projection")
for iteration in range(training_iters):
    batch_inputs, batch_targets=next_batch(mini_batch_size, train_images, train_labels)
    feed_dict_mini_batch={input_realisation_network: train_images[:realisation_batch_size], input_error_calculation_network: batch_inputs, target_labels: batch_targets, keep_prob_ph: dropout_keep_probs_training, ph_use_identical_patterns_across_ensemble: False, is_train: True}
    if (iteration%50)==0: 
        cumul_cpu_time+=time.time()-start_time
        lossv,accuracyv=sess.run([loss_mean, accuracy],feed_dict=feed_dict_mini_batch)
        epoch=iteration*mini_batch_size/len(train_images)
        print(dataset_name,use_target_space,use_adam,learning_rate, realisation_batch_size,pseudoinverse_regularisation_constant,mini_batch_size,iteration,round(epoch,4),"tr",lossv,accuracyv,end=',',sep=",")
        # we split the test set into chunks to save on memory.  Set split_test_set_into_chunks=1 if you're not worried about memory.
        chunk_size1=test_images1.shape[0]//split_test_set_into_chunks
        chunk_size2=test_images2.shape[0]//split_test_set_into_chunks
        test_loss_mean=[[]]*split_test_set_into_chunks
        test_acc_mean=[[]]*split_test_set_into_chunks
        valid_loss_mean=[[]]*split_test_set_into_chunks
        valid_acc_mean=[[]]*split_test_set_into_chunks
        for chunk in range(split_test_set_into_chunks):
            #print("chunk",chunk)
            [valid_loss_mean[chunk],valid_acc_mean[chunk]]=sess.run([loss_mean,accuracy], feed_dict={input_realisation_network: train_images[:realisation_batch_size], input_error_calculation_network: test_images1[chunk*chunk_size1:(chunk+1)*chunk_size1], target_labels: test_labels1[chunk*chunk_size1:(chunk+1)*chunk_size1], keep_prob_ph: dropout_keep_probs_testing, ph_use_identical_patterns_across_ensemble:True, is_train: False})        
            [test_loss_mean[chunk],test_acc_mean[chunk]]= sess.run([loss_mean,accuracy], feed_dict={input_realisation_network: train_images[:realisation_batch_size], input_error_calculation_network: test_images2[chunk*chunk_size2:(chunk+1)*chunk_size2], target_labels: test_labels2[chunk*chunk_size2:(chunk+1)*chunk_size2], keep_prob_ph: dropout_keep_probs_testing, ph_use_identical_patterns_across_ensemble:True, is_train: False})
        
        test_loss_mean=np.stack(test_loss_mean).mean()
        test_acc_mean=np.stack(test_acc_mean).mean()
        valid_loss_mean=np.stack(valid_loss_mean).mean()
        valid_acc_mean=np.stack(valid_acc_mean).mean()

        print("vld", valid_loss_mean,valid_acc_mean,"test",test_loss_mean,test_acc_mean, round(cumul_cpu_time,4),("double" if use_double_keep_prob else "")+("realisation" if use_realisation_keep_prob else "")+str(training_keep_prob).replace(" ","").replace(",","-"), (target_initialiser if use_target_space else weight_initialiser),afs,str(fully_connected_hidden_nodes_per_layer).replace(" ","").replace(",","-"), str([conv_kernel_side_length_per_layer, max_pool_side_length_per_layer, channels_per_layer]).replace(" ","").replace(",","-"), str(batch_normalisation_layers).replace(" ","").replace(",","-"),do_initial_targets_projection,sep=",")
        start_time = time.time()
    sess.run(update, feed_dict=feed_dict_mini_batch)
sess.close()
