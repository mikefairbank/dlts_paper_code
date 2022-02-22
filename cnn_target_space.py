# This shows an example of how a CNN can be built in target space.  
# This code is to accompany the paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L. Citi, https://jmlr.org/papers/v23/20-040.html 
# Repository: https://github.com/mikefairbank/dlts_paper_code
# Note that the experiments in the paper used the TF1 version of this code (provided in the same repository)
# Please cite the above paper if this code or future variants of it are used in future academic work.

# To run on the cifar10 dataset, use:
# python3 cnn_target_space.py --weights --max_epoch 400 --mbs 100 --realisation_batch_size 100 --prc 0.1  --dataset cifar10
# python3 cnn_target_space.py --targets --max_epoch 40 --mbs 100 --realisation_batch_size 100 --prc 0.1 --dataset cifar10

import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ts_layers as ts # This contains the main target-space program logic
import sys
import time

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--targets', action='store_true', help='Use target space')
parser.add_argument('--weights', action='store_true', help='Use weight space')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--sgd', action='store_true',  help='Use SGD (as opposed to Adam)')
parser.add_argument('--mbs', type=int, default=10, help='mini batch size (default 10)')
parser.add_argument('--lr', type=float, help='learning_rate')
parser.add_argument('--max_its', type=int, help='max_its')
parser.add_argument('--max_epoch', type=int, default=0,help='max_epoch')
parser.add_argument('--realisation_batch_size', type=int, default=100, help='realisation_batch_size')
parser.add_argument('--prc', type=float, default=0.1, help="pseudoinverse_regularisation_constant")
parser.add_argument('--avoid_projection', action='store_true', help='avoid target initial projection')
parser.add_argument('--dataset', type=str, default="mnist", help='dataset cifar100/mnist/cifar10/fashion')
parser.add_argument('--log_results', action='store_true', help='Log results in csv format')
parser.add_argument('--steps_per_epoch',type=int)

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


training_set_size=train_images.shape[0]

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



use_target_space=args.targets or not(args.weights)
learning_rate=args.lr if args.lr!=None else (0.1 if args.sgd else 0.001)
pseudoinverse_regularisation_constant=args.prc 
mini_batch_size = 10 if args.mbs==None else args.mbs
realisation_batch_size=args.realisation_batch_size
af=layers.LeakyReLU(alpha=0.2)
steps_per_epoch=args.steps_per_epoch if args.steps_per_epoch!=None else training_set_size//mini_batch_size

if use_target_space:
    # build FFNN with CNN architecture, in target space
    class TSModel(keras.Model):
        def __init__(self):
            super(TSModel, self).__init__()
            self.fixed_targets_input_matrix=tf.constant(train_images[:realisation_batch_size],tf.float32)
            self.tslayers=[]
            
            self.tslayers.append(ts.TSConv2D(realisation_batch_size=realisation_batch_size, pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant, filters=32, kernel_size=3, activation=af, padding="same"))
            self.tslayers.append(ts.TSConv2D(realisation_batch_size=realisation_batch_size, pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant, filters=32, kernel_size=3, activation=af, padding="same"))
            self.tslayers.append(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
            if args.dropout:
                self.tslayers.append(layers.Dropout(0.2))
            self.tslayers.append(ts.TSConv2D(realisation_batch_size=realisation_batch_size, pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant, filters=64, kernel_size=3, activation=af, padding="same"))
            self.tslayers.append(ts.TSConv2D(realisation_batch_size=realisation_batch_size, pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant, filters=64, kernel_size=3, activation=af, padding="same"))
            self.tslayers.append(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
            if args.dropout:
                self.tslayers.append(layers.Dropout(0.2))
            self.tslayers.append(ts.TSConv2D(realisation_batch_size=realisation_batch_size, pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant, filters=128, kernel_size=3, activation=af, padding="same"))
            self.tslayers.append(ts.TSConv2D(realisation_batch_size=realisation_batch_size, pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant, filters=128, kernel_size=3, activation=af, padding="same"))
            self.tslayers.append(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
            if args.dropout:
                self.tslayers.append(layers.Dropout(0.2))
            self.tslayers.append(keras.layers.Flatten())
            self.tslayers.append(ts.TSDense(128, realisation_batch_size=realisation_batch_size,  activation=af, name='dense_1', pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant))
            self.tslayers.append(ts.TSDense(num_classification_categories,  realisation_batch_size=realisation_batch_size, name='output',  pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant))

            
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
    #print("test output",keras_model(train_images[:11]))
    if not args.avoid_projection:
        test_output=keras_model(train_images[:11]) # annoyingly we have to do this once to force the target matrices to all be built.
        # It can't build the target layers previously because it doesn't know the image width/height that is propagated through the later layers.
        keras_model.initialise_target_layers_with_projection()

else:
    # build FFNN with CNN architecture, in weight space
    inputs = keras.Input(shape=(input_image_side_length,input_image_side_length,input_image_channels,), name='input')
    #x = layers.Conv2D(32, kernel_size=3, activation='relu', padding="same")(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation=af, padding="same")(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation=af, padding="same")(inputs)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    if args.dropout:
        x=layers.Dropout(0.2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation=af, padding="same")(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation=af, padding="same")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    if args.dropout:
        x=layers.Dropout(0.2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation=af, padding="same")(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation=af, padding="same")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    if args.dropout:
        x=layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=af, name='dense_1')(x)
    outputs = layers.Dense(num_classification_categories, name='output')(x)
    keras_model = keras.Model(inputs=inputs, outputs=outputs)


if args.sgd:
    optimizer=keras.optimizers.SGD(learning_rate)
else:
    optimizer=keras.optimizers.Adam(learning_rate)

if args.log_results:
    print("targets,iter,epoch,accuracy,test_accuracy,mbs,rbs,learning_rate,steps_per_epoch,prc,cumul_cpu_time")
    start_time = time.time()
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            iteration=(epoch+1)*args.steps_per_epoch
            num_training_patterns_seen=iteration*mini_batch_size
            true_epoch=num_training_patterns_seen/training_set_size
            print(use_target_space,iteration,round(true_epoch,3),round(logs["sparse_categorical_accuracy"],4),round(logs["val_sparse_categorical_accuracy"],4),mini_batch_size,realisation_batch_size,learning_rate,steps_per_epoch,args.prc,round(time.time()-start_time,4),sep=",")
    callbacks=CustomCallback()
    verbose=0
else:
    callbacks=None
    verbose=1
    

keras_model.compile(optimizer=optimizer,  
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
              
             
history = keras_model.fit(train_images, train_labels,
                batch_size=args.mbs,
                epochs=(args.max_epoch*training_set_size+1)//(mini_batch_size*steps_per_epoch),
                steps_per_epoch=steps_per_epoch,
                validation_data=(test_images, test_labels),validation_freq=1,
                verbose=verbose, callbacks=callbacks)

