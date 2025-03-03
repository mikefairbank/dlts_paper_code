# Example code using bespoke Keras layers to implemente a dense NN in target space.
# This code is to accompany the paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L. Citi, https://jmlr.org/papers/v23/20-040.html 
# Repository: https://github.com/mikefairbank/dlts_paper_code
# Note that the experiments in the paper used the TF1 version of this code (provided in the same repository)
# Please cite the above paper if this code or future variants of it are used in future academic work.
#
# Example, to run code from command line, use either:
# python3 twoSpirals_target_space.py --targets --max_its 4000 --lr 10.0 --graphical 
# python3 twoSpirals_target_space.py --targets --max_its 1000 --adam --graphical 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ts_layers as ts # This contains the main target-space program logic
import time 

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--targets', action='store_true', help='Use target space')
parser.add_argument('--weights', action='store_true', help='Use weight space')
parser.add_argument('--adam', action='store_true',  default=False, help='Use Adam')
parser.add_argument('--lr', type=float, help='learning_rate')
parser.add_argument('--max_its', type=int, default=4000, help='max_its')
parser.add_argument('--avoid_projection', action='store_true', help='avoid target initial projection')
parser.add_argument('--graphical', action='store_true')
parser.add_argument('--screenshot', action='store_true')
parser.add_argument('--prc', type=float, default=0.001,  help='pseudoinverse amount of L2 regularisation')
#parser.add_argument('--target_initialiser', type=float, default=1,  help='magnitude of initialised target matrices')

args = parser.parse_args()


use_target_space=args.targets or not(args.weights)
use_adam=args.adam
learning_rate=args.lr if args.lr!=None else (0.01 if use_adam else 0.1)
max_its=args.max_its
graphical=args.graphical

pseudoinverse_regularisation_constant=args.prc
#target_initialiser=args.target_initialiser

train_inputs=pd.read_csv('datasets/twoSpirals.csv',usecols = [0,1],skiprows = None,header=None).values.astype(np.float32)
train_outputs = pd.read_csv('datasets/twoSpirals.csv',usecols = [2],skiprows = None ,header=None).values.reshape([-1])
test_inputs=pd.read_csv('datasets/twoSpiralsTestSet.csv',usecols = [0,1],skiprows = None,header=None).values.astype(np.float32)
test_outputs = pd.read_csv('datasets/twoSpiralsTestSet.csv',usecols = [2],skiprows = None ,header=None).values.reshape([-1])
realisation_batch_size=len(train_inputs)

if use_target_space:
    # build FFNN with architecture 2-12-12-12-2, in target space

    # On the first layer, the target_input_matrix should be a representative batch of inputs (and this may be different from 
    # the training inputs; but in this case not.) For further details see section 3.1 of the target space paper.
    # When building the FFNN with target space, we need to make sure that each "target_input_matrix" argument receives 
    # a pointer to the previous layer's "target_output_matrix".  This allows the target-space "sequential cascade untangling (SCU)" algorithm to take place.
    class TSModel(keras.Model):
        def __init__(self):
            super(TSModel, self).__init__()
            self.fixed_targets_input_matrix=tf.constant(train_inputs,tf.float32)
            self.tslayers=[]
            
            self.tslayers.append(ts.TSDense(12, realisation_batch_size=realisation_batch_size, activation='tanh', name='dense_1',pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant))
            self.tslayers.append(ts.TSDense(12, realisation_batch_size=realisation_batch_size, activation='tanh', name='dense_2', pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant))
            self.tslayers.append(ts.TSDense(12, realisation_batch_size=realisation_batch_size, activation='tanh', name='dense_3', pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant))
            self.tslayers.append(ts.TSDense(2, realisation_batch_size=realisation_batch_size, name='output', pseudoinverse_l2_regularisation=pseudoinverse_regularisation_constant))
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
    # build FFNN with architecture 2-12-12-12-2, in weight space
    inputs = keras.Input(shape=(2,), name='input')
    x = layers.Dense(12, activation='tanh', name='dense_1')(inputs)
    x = layers.Dense(12, activation='tanh', name='dense_2')(x)
    x = layers.Dense(12, activation='tanh', name='dense_3')(x)
    outputs = layers.Dense(2, name='output')(x)
    keras_model = keras.Model(inputs=inputs, outputs=outputs)



#if use_target_space and not not args.avoid_projection:
#    do_initial_targets_projection()

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


class CallbackUpdateGraphics(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.count = 0

    def on_batch_end(self, batch, logs):
        self.count+=1
        if self.count%40==0:
            y_greyscale=tf.nn.softmax(keras_model(image_input_matrix),axis=1)[:,0].numpy()
            y_greyscale=y_greyscale.reshape(graphics_resolution+1,graphics_resolution+1).T
            myobj.set_data(y_greyscale)
            plt.show()
            plt.pause(0.001)
            print("iterations",self.count)
            
callbacks = [CallbackUpdateGraphics()] if graphical else []

if args.adam:
    optimizer=keras.optimizers.Adam(learning_rate)
else:
    optimizer=keras.optimizers.SGD(learning_rate)

keras_model.compile(optimizer=optimizer,  
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
start_time = time.time()

history = keras_model.fit(train_inputs, train_outputs,
                batch_size=len(train_inputs),
                epochs=max_its,
                validation_data=(test_inputs, test_outputs),validation_freq=400,
                verbose=0 if args.graphical else 1,
                callbacks=callbacks)
end_time = time.time()
print("Training time:",end_time-start_time)

if graphical:
    if args.screenshot:
        import datetime
        plt.savefig("trained_net_"+("t" if use_target_space else "w")+"_"+str(datetime.datetime.now())+".png",bbox_inches="tight")
    else:
        input("Press [enter] to continue.")
