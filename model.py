import theano
from theano import tensor as T
import numpy as np
import backend as K
import optimizer
import regularizers
import layers
import net
import time
import math
import cPickle as pkl
import sys
from scipy import io
sys.setrecursionlimit(2000)

class Model():

    def __init__(self, nets=[]):
        self.nets = []
        for net in nets:
            self.add(net)

    def add(self, net):
        self.nets.append(net)

    def connect(self,net_pre, net_aft):
        net_aft.layers[0].set_previous(net_pre.layers[-1])

    @property
    def updates(self):
        updates = []
        for net in self.nets:
            if hasattr(net,'updates'):
                updates += net.updates
        return updates

    @property
    def params(self):
        params = []
        for net in self.nets:
            if hasattr(net,'params'):
                params += net.params
        return params

    @property
    def regularizers(self):
        regularizers = []
        for net in self.nets:
            regularizers += net.regularizers
        return regularizers

    def get_out(self,train=True):
        outs = []
        for net_i in xrange(len(self.nets)):
            outs.append(self.nets[net_i].get_out(train=train))
        return outs[-1]


    def compile(self, options):
        '''Configure the learning process.
        '''
        # input of model
        self.X = T.tensor3(name='input_frames', dtype='float32')      

        self.init_h = T.matrix(name='input_hidden', dtype='float32')
        self.init_m = T.matrix(name='input_memory', dtype='float32')

        netlu = self.nets[0]
        netru = self.nets[1]
        netrv = self.nets[2]
        netrm = self.nets[3]

        netlu.set_input([self.init_h,self.init_m])

        idx = 0
        for l in netlu.layers:
            if hasattr(l, 'has_input_frame'):
                if l.has_input_frame:
                    l.input_frame = self.X[:,idx,:]
                    idx += 1
        assert  idx == options['v_length']


        def comp_(train):
            netlu.set_out(train=train)

            if not train:
                [my_H, my_M] = netlu.get_out_idx(-2)
                print 'compile encoder...'
                self._encoder = theano.function([self.X,self.init_h,self.init_m], my_H)

            self.y_pred = netru.get_out(train=train)
            assert len(self.y_pred) == options['v_length']
            loss_backward = T.sum(T.sqr(self.X[:,-1,:] - self.y_pred[0]))
            for i in xrange(1,options['v_length']):
                loss_backward += T.sum(T.sqr(self.X[:,-1-i,:] - self.y_pred[i]))

            self.y_pred2 = netrv.get_out(train=train)
            assert len(self.y_pred2) == options['v_length']
            loss_forward = T.sum(T.sqr(self.X[:,0,:] - self.y_pred2[0]))
            for i in xrange(1,options['v_length']):
                loss_forward += T.sum(T.sqr(self.X[:,i,:] - self.y_pred2[i]))
            
            self.y_mean = netrm.get_out(train=train)
            assert len(self.y_mean) == 1
            loss_mean= 200. * T.sum(T.sqr(T.mean(self.X, axis=1) - self.y_mean[0]))
            whts = options['weights']
            loss = whts[0]*loss_backward + whts[1]*loss_forward + whts[2]*loss_mean            
            for r in self.regularizers:
                loss = r(loss)
            
            if train:
                self.optimizer = eval('optimizer.'+ options['optimizer'])(self.params, lr=options['lrate'])
                updates = self.optimizer.get_updates(self.params, loss)
                updates += self.updates
                print 'compile train...'
                start_time = time.time()
                self._train = theano.function([self.X,self.init_h,self.init_m], loss, updates=updates)
                end_time = time.time()
                print 'spent %f seconds'  % (end_time-start_time)
            else:
                print 'compile test...'
                start_time = time.time()
                self._test = theano.function([self.X,self.init_h,self.init_m], loss)
                end_time = time.time()
                print 'spent %f seconds'  % (end_time-start_time)

        comp_(train=True)
        comp_(train=False)



    def train(self, train_data, test_data, options):

        validFreq = options['validFreq']
        saveFreq = options['saveFreq']
        dispFreq = options['dispFreq']
        max_iter = options['max_iter']
        saveto =options['saveto']

        train_loss_his = []
        test_loss_his = []

        start_time = time.time()

        test_loss_ = self.test_loss(self._test, test_data, options)
        test_loss_his.append(test_loss_)
        print 'Valid cost:', test_loss_

        train_loss = 0.

        try:
            for uidx in xrange(1,max_iter+1):

                x = train_data.GetBatch()
                train_loss = self._train(x,np.zeros((x.shape[0],options['dim_proj']),dtype=np.float32),np.zeros((x.shape[0],options['dim_proj']),dtype=np.float32))

                if np.isnan(train_loss) or np.isinf(train_loss):
                    print 'bad cost detected: ', train_loss

                if np.mod(uidx, dispFreq) == 0 or uidx is 1:
                    train_loss = train_loss/(x.shape[0]*x.shape[1])
                    train_loss_his.append(train_loss)
                    print 'Step ', uidx,  'Train cost:', train_loss

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print 'Saving...',
                    params_to_save = self.get_params_value()
                    updates_value = self.get_updates_value()
                    np.savez(saveto, params=params_to_save, updates_v=updates_value, train_loss_his=train_loss_his, test_loss_his=test_loss_his)
                    pkl.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Save Done'

                if np.mod(uidx, validFreq) == 0:

                    valid_loss_ = self.test_loss(self._test, test_data, options)
                    test_loss_his.append(valid_loss_)
                    print 'Valid cost:', valid_loss_

        except KeyboardInterrupt:
            print "Training interupted"

        if saveto:
            print 'Saving...',
            params_to_save = self.get_params_value()
            updates_value = self.get_updates_value()
            np.savez(saveto, params=params_to_save, updates_v=updates_value, train_loss_his=train_loss_his, test_loss_his=test_loss_his)
            pkl.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
            print 'Save Done'

        end_time = time.time()
        print  ('Training took %.1fs' % (end_time - start_time))


    def test(self, test_data, options):

        max_iter = int(math.ceil(float(test_data.data_size_)/float(test_data.batch_size_)))
        start_time = time.time()

        test_loss_ = self.test_loss(self._test, test_data, options)
        print 'Valid cost:', test_loss_

        try:
            for uidx in xrange(1,max_iter+1):
                x = test_data.GetBatch()
                hidden_orig_cpu = self._encoder(x,np.zeros((x.shape[0],options['dim_proj']),dtype=np.float32),np.zeros((x.shape[0],options['dim_proj']),dtype=np.float32))
                io.savemat('results/hidden_' + str(uidx) + '.mat', {'hidden': hidden_orig_cpu})

        except KeyboardInterrupt:
            print "Test interupted"

        end_time = time.time()
        print  ('Testing took %.1fs' % (end_time - start_time))


    def test_loss(self, f_pred, data, options):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        """
        pred_loss = 0.
        sum_iter = int(math.ceil(float(data.data_size_)/float(data.batch_size_)))
        for i in range(sum_iter):
            x = data.GetBatch()
            pred_loss += f_pred(x,np.zeros((x.shape[0],options['dim_proj']),dtype=np.float32),np.zeros((x.shape[0],options['dim_proj']),dtype=np.float32))
        pred_loss = pred_loss/(data.data_size_*data.seq_length_)
        return pred_loss

    def get_params_value(self):
        new_params = [par.get_value() for par in self.params]
        return new_params

    def get_updates_value(self):
        updates = [par.get_value() for (par,up) in self.updates]
        return  updates

    def reload_params(self, params_file):
        print 'Reloading model params'
        ff = np.load(params_file)
        new_parms = ff['params']
        for idx, par in enumerate(self.params):
            K.set_value(par, new_parms[idx])
        new_updates = ff['updates_v']
        for idx, (par,up) in enumerate(self.updates):
            K.set_value(par, new_updates[idx])