import cPickle as pkl
import sys
import time
import numpy

from net import *
from model import *
from layers import *
import DataHelper


def run_blstm(
    dim_proj=256,  # LSTM number of hidden units.
    dim_frame=4096, # feature dimension of image frame in the video
    v_length = 25, # video length or number of frames
    max_iter=20000,  # The maximum number of epoch to run  
    l2_decay=0.0001,  # Weight decay for model params.
    lrate=0.001,  # Learning rate for SGD, Adam
    optimizer='SGD',  # SGD, Adam available
    saveto='blstm_model.npz',  # The best model will be saved there
    dispFreq=10,  # Display to stdout the training progress every N updates
    validFreq=1000,  # Compute the validation error after this number of update.
    saveFreq=1000,  # Save the parameters after every saveFreq updates
    batch_size=100,  # The batch size during training.
    valid_batch_size=100,  # The batch size used for validation/test set.
    weights=[1./3.,1./3.,1./3.],  # The Weights for forwoad and backward reconstruction and mean value reconstruction
    reload_model=False,  # If reload model from saveto.
    is_train=True
):
    model_options = locals().copy()
    if reload_model:
        print "Reloading model options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)
    print "model options", model_options

    print 'Loading data'
    if is_train:
    	train_data = DataHelper.DataHelper(model_options['v_length'], model_options['batch_size'],model_options['dim_frame'],
    		data_file= './data/fcv_train_feats.h5', train=True) 
    test_data = DataHelper.DataHelper(model_options['v_length'], model_options['valid_batch_size'],model_options['dim_frame'],
            data_file= './data/fcv_test_feats.h5', train=False)
    
    # Encoder LSTMs
    net_lu = Net()
    lstmpar = LstmParams(model_options['dim_proj'], model_options['dim_frame'])
    for i in xrange(model_options['v_length']):
        net_lu.add(LSTM_Unit(2*i, lstmpar, model_options['batch_size'], model_options['dim_proj'], model_options['dim_frame'], model_options['l2_decay']))
        net_lu.add(BinaryLayer(2*i+1))

   
    # Backword Decoder LSTMs
    net_ru = Net()
    decpar = DecLstmParams(model_options['dim_proj'], model_options['dim_frame'])
    for i in xrange(model_options['v_length']):
        net_ru.add(LSTM_Dec(i, decpar, model_options['batch_size'], model_options['dim_proj'], model_options['dim_frame'], model_options['l2_decay']))

    # Forword Decoder LSTMs
    net_rv = Net()
    decpar2 = DecLstmParams(model_options['dim_proj'], model_options['dim_frame'])
    for i in xrange(model_options['v_length']):
        net_rv.add(LSTM_Dec(i, decpar2, model_options['batch_size'], model_options['dim_proj'], model_options['dim_frame'], model_options['l2_decay']))

   
    # Other Decoder LSTMs
    net_rm = Net()
    decpar3 = DecLstmParams(model_options['dim_proj'], model_options['dim_frame'])
    net_rm.add(LSTM_Dec(0, decpar3, model_options['batch_size'], model_options['dim_proj'], model_options['dim_frame'], model_options['l2_decay']))

    
    model = Model()
    model.add(net_lu)
    model.add(net_ru)
    model.add(net_rv)
    model.add(net_rm)
    model.connect(net_lu, net_ru)
    model.connect(net_lu, net_rv)
    model.connect(net_lu, net_rm)

    if reload_model:
        model.reload_params(saveto)

    model.compile(model_options)


    if is_train:
        model.train(train_data, test_data, model_options)
    else:
        model.test(test_data, model_options)

    
if __name__ == '__main__':
    # is_train:
    #   True for training model,
    #   False for testing model and generate hidden vectors for test data.
    run_blstm(is_train=True)

