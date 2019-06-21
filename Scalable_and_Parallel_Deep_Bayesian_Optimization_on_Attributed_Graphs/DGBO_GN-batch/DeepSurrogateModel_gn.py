

#=========================================
#We need to implement a class with two functions including
#train() and get_basis(), where train() function is to
#train the deep model and get_basis() function is to get
#the basis (or called embedding, latent representation) of x_test
#=========================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import graph_nets as gn
from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets.demos import models
from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf
import scipy.sparse as sp

try:
    import seaborn as sns
except ImportError:
    pass
else:
    sns.reset_orig()


EDGES=1
NODES=0
GLOBAL=4
N_NODE=5
N_EDGE=6

class Deep_GraphNet_Process():

    def __init__(self,block_num=1,block_elayer_num=5,block_nlayer_num=5,block_glayer_num=5,block_elayer_size=89,block_nlayer_size=89,block_glayer_size=89,block_e_act=tf.nn.tanh,block_n_act=tf.nn.tanh,block_g_act=tf.nn.tanh,encoder_elayer_num=5,encoder_nlayer_num=5,encoder_glayer_num=5,encoder_elayer_size=89,encoder_nlayer_size=89,encoder_glayer_size=89,encoder_e_act=tf.nn.tanh,encoder_n_act=tf.nn.tanh,encoder_g_act=tf.nn.tanh,pooling_e_size=10,pooling_n_size=10,pooling_g_size=10,fc_num=5,fc_size=91,fc_act=tf.nn.tanh, learning_rate=1e-4,weight_decay=1e-5,epochs=800,batchsize=None):
        self.info=None
        self.layers=[]
        self.block_num=block_num
        self.block_elayer_num=block_elayer_num
        self.block_nlayer_num=block_nlayer_num
        self.block_glayer_num=block_glayer_num
        self.block_elayer_size=block_elayer_size
        self.block_nlayer_size=block_nlayer_size
        self.block_glayer_size=block_glayer_size
        self.block_e_act=block_e_act
        self.block_n_act=block_n_act
        self.block_g_act=block_g_act
        self.encoder_elayer_num=encoder_elayer_num
        self.encoder_nlayer_num=encoder_nlayer_num
        self.encoder_glayer_num=encoder_glayer_num
        self.encoder_elayer_size=encoder_elayer_size
        self.encoder_nlayer_size=encoder_nlayer_size
        self.encoder_glayer_size=encoder_glayer_size
        self.encoder_e_act=encoder_e_act
        self.encoder_n_act=encoder_n_act
        self.encoder_g_act=encoder_g_act
        self.pooling_e_size=pooling_e_size
        self.pooling_n_size=pooling_n_size
        self.pooling_g_size=pooling_g_size
        self.fc_num=fc_num
        self.fc_size=fc_size
        self.fc_act=fc_act
    
        self.model=None
        self.sess=None
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.batchsize=batchsize
    
        self.input_ph=None
        self.target_ph=None
        self.ph_list=None
    
        self.loss=None
        
        if self.batchsize==None:
            self.fullbatch=True
        else:
            self.fullbatch=False
    
        #self.initializers={"w": tf.truncated_normal_initializer(stddev=1.0),"b": tf.truncated_normal_initializer(stddev=1.0)}
        #self.initializers={"w": tf.glorot_uniform_initializer(),"b": tf.zeros_initializer()}
        #self.regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=self.weight_decay),"b": tf.contrib.layers.l2_regularizer(scale=self.weight_decay)}
    

    
    
    def train(self, X, y, flag):
        ##add code to train this deep model
        input_graphs_for_training=X
        #print("haha:",X[0].graph)
        if flag == True:
            
            if self.sess!=None:
                self.sess.close()

            #clear and reset the default graph
            tf.reset_default_graph()

            if self.fullbatch==True:
                self.batchsize=len(input_graphs_for_training)
            print
            #input and target placeholders
            input_ph, target_ph=self.create_placeholders(input_graphs_for_training,batchsize=self.batchsize)
            transition_matrix_e=tf.placeholder(tf.float64, shape=(None,None))
            transition_matrix_n=tf.placeholder(tf.float64, shape=(None,None))
            transition_matrix_g=tf.placeholder(tf.float64, shape=(None,None))
            ph_list=[transition_matrix_e,transition_matrix_n,transition_matrix_g]
            self.input_ph=input_ph
            self.target_ph=target_ph
            self.ph_list=ph_list
            

            
            #print("input ph:",input_ph)
            #create the network
            self.model=Deep_GraphNet(self,ph_list)
            output_pre=self.model(input_ph)
            #create loss
            loss=self.model.get_loss()
            self.loss=loss
            #optimizer
            optimizer=tf.train.AdamOptimizer(self.learning_rate)
            step_op=optimizer.minimize(loss)
            #create the session and initialize
            self.sess=tf.Session()
            self.sess.run(tf.global_variables_initializer())
            #training
            start_train_time=time.time()
            canidx_list=list(np.arange(len(input_graphs_for_training)))
            for iter in range(self.epochs):
                start_epoch_time=time.time()
                #np.random.shuffle(canidx_list)
                batchstartidx=0
                loss_restore=[]
                while batchstartidx<len(X):
                    feed_dict = self.create_feed_dict(input_graphs_for_training,y,self.input_ph,target_ph,ph_list,batchsize=self.batchsize,canidx_list=canidx_list,batchstartidx=batchstartidx)
                    train_values = self.sess.run({"setp":step_op,"target":target_ph,"loss":[loss],"outputs":[output_pre]},feed_dict=feed_dict)
                    batchstartidx+=self.batchsize
                    #print("echo: ",iter,", batch size: ",self.batchsize,", loss: ",train_values["loss"])
                    loss_restore.append(train_values["loss"][0])
                end_epoch_time=time.time()
                if (iter+1)%100==0:
                    print(iter+1,"/",self.epochs,", loss : ",min(loss_restore),", accumulative training time: ",end_epoch_time-start_train_time,"s")
        feed_dict = self.create_feed_dict(input_graphs_for_training,y,self.input_ph,self.target_ph,self.ph_list,batchsize=len(input_graphs_for_training),canidx_list=list(np.arange(len(input_graphs_for_training))),batchstartidx=0)
        get_values = self.sess.run({"loss":[self.loss],"basis":self.model.basis},feed_dict=feed_dict)
        print("loss=",get_values["loss"][0])
        #print(get_values["basis"],get_values["basis"].shape)
        return get_values["basis"]

    def get_basis(self, X_test, y_test=[]):
        ##add code to get the basis from the deep model according to the X_test
        feed_dict = self.create_feed_dict(X_test,y_test,self.input_ph,self.target_ph,self.ph_list,batchsize=len(X_test),canidx_list=list(np.arange(len(X_test))),batchstartidx=0)
        if len(y_test)==0:
            get_values = self.sess.run({"basis":self.model.basis},feed_dict=feed_dict)
        else:
            get_values = self.sess.run({"acc":[self.model.get_accuracy()],"loss":[self.loss],"basis":self.model.basis,"outputs":self.model.outputs},feed_dict=feed_dict)
            print("acc :",get_values["acc"][0],"loss=",get_values["loss"][0])
            #self.sess.close()
        return get_values["basis"]
    
    def create_placeholders(self,input_graphs,batchsize=20):
        #input_ph = utils_tf.placeholders_from_networkxs(list(input_graphs[0:batchsize]), force_dynamic_num_graphs=True)
        input_ph = utils_tf.placeholders_from_networkxs([input_graphs[0]], force_dynamic_num_graphs=True)
        output_ph = tf.placeholder(tf.float64, shape=(None,1))
        return input_ph, output_ph
    
    def create_feed_dict(self, input_graphs,targets,input_ph,target_ph,ph_list,batchsize=20,canidx_list=[],batchstartidx=0):
        if (batchstartidx+batchsize) > len(canidx_list):
            batch_list=canidx_list[batchstartidx:batchstartidx+batchsize]+canidx_list[0:batchstartidx+batchsize-len(canidx_list)]
        else:
            batch_list=canidx_list[batchstartidx:batchstartidx+batchsize]
        #print("batch_list:",batch_list)
        inputs = utils_np.networkxs_to_graphs_tuple([input_graphs[ii] for ii in batch_list])
        if len(targets)!=0:
            outputs = np.array([targets[ii] for ii in batch_list]).reshape(len(batch_list),1)
        else:
            outputs = np.array([]).reshape(0,1)
        ##create the transition_matrix of nodes, edges, and global
        
        transition_matrix_e=np.zeros(inputs[N_EDGE].shape[0]*sum(inputs[N_EDGE])).reshape(inputs[N_EDGE].shape[0],sum(inputs[N_EDGE]))
        start_idx=0
        for i in range(inputs[N_EDGE].shape[0]):
            transition_matrix_e[i,start_idx:start_idx+inputs[N_EDGE][i]]=1
            start_idx+=inputs[N_EDGE][i]

        transition_matrix_n=np.zeros(inputs[N_NODE].shape[0]*sum(inputs[N_NODE])).reshape(inputs[N_NODE].shape[0],sum(inputs[N_NODE]))
        start_idx=0
        for i in range(inputs[N_NODE].shape[0]):
            transition_matrix_n[i,start_idx:start_idx+inputs[N_NODE][i]]=1
            start_idx+=inputs[N_NODE][i]
        transition_matrix_g=np.eye(batchsize)
        """
        ##sparse
        ones_idx_row=[]
        ones_idx_col=[]
        start_idx=0
        for i in range(inputs[N_EDGE].shape[0]):
            ones_idx_row=ones_idx_row + [ int(i) for _ in range(inputs[N_EDGE][i]) ]
            ones_idx_col=ones_idx_col + [ int(start_idx+j) for j in range(inputs[N_EDGE][i])]
            start_idx+=inputs[N_EDGE][i]
        transition_matrix_e=sp.coo_matrix((np.ones(sum(inputs[N_EDGE])), (ones_idx_row, ones_idx_col)), shape=(inputs[N_EDGE].shape[0],sum(inputs[N_EDGE])))

        ones_idx_row=[]
        ones_idx_col=[]
        start_idx=0
        for i in range(inputs[N_NODE].shape[0]):
            ones_idx_row=ones_idx_row + [ int(i) for _ in range(inputs[N_NODE][i]) ]
            ones_idx_col=ones_idx_col + [ int(start_idx+j) for j in range(inputs[N_NODE][i])]
            start_idx+=inputs[N_NODE][i]
        transition_matrix_n=sp.coo_matrix((np.ones(sum(inputs[N_NODE])), (ones_idx_row, ones_idx_col)), shape=(inputs[N_NODE].shape[0],sum(inputs[N_NODE])))

        transition_matrix_g=sp.eye(inputs[N_NODE].shape[0])
        """
        feed_dict = {input_ph: inputs, target_ph: outputs, ph_list[0]:transition_matrix_e, ph_list[1]:transition_matrix_n, ph_list[2]:transition_matrix_g}

        return feed_dict

    """
    def make_mlp_model_with_layernorm(self,layer_num, layer_size, layer_act,name="mlp"):
        ###according to https://arxiv.prg/pdf/1607.06450.pdf
        return snt.Sequential([snt.nets.MLP([layer_size] * layer_num, activation=layer_act, activate_final=True,initializers={"w": tf.glorot_uniform_initializer(),"b": tf.zeros_initializer()},regularizers={"w": tf.contrib.layers.l2_regularizer(scale=self.weight_decay),"b": tf.contrib.layers.l2_regularizer(scale=self.weight_decay)},name=name), snt.LayerNorm()])
    """
    def make_mlp_model_with_layernorm(self,layer_num, layer_size, layer_act,name="mlp"):
        return snt.nets.MLP([layer_size] * layer_num, activation=layer_act, activate_final=True,initializers={"w": tf.glorot_uniform_initializer(),"b": tf.zeros_initializer()},regularizers={"w": tf.contrib.layers.l2_regularizer(scale=self.weight_decay),"b": tf.contrib.layers.l2_regularizer(scale=self.weight_decay)},name=name)

class Deep_GraphNet(snt.AbstractModule):
    def __init__(self,DGN_Process,ph_list,name="Deep_GraphNet"):
        super(Deep_GraphNet, self).__init__(name=name)
        self.layers=[]
        self.DGN_Process=DGN_Process
        self.ph_list=ph_list
        
        self.encoder = gn.modules.GraphIndependent(
        #self.encoder = gn.modules.GraphNetwork(
                                            edge_model_fn=lambda: self.DGN_Process.make_mlp_model_with_layernorm(self.DGN_Process.encoder_elayer_num,self.DGN_Process.encoder_elayer_size,self.DGN_Process.encoder_e_act,name="mlp_in_encoder_e"),
                                            node_model_fn=lambda: self.DGN_Process.make_mlp_model_with_layernorm(self.DGN_Process.encoder_nlayer_num,self.DGN_Process.encoder_nlayer_size,self.DGN_Process.encoder_n_act,name="mlp_in_encoder_n"),
                                            global_model_fn=lambda: self.DGN_Process.make_mlp_model_with_layernorm(self.DGN_Process.encoder_glayer_num,self.DGN_Process.encoder_glayer_size,self.DGN_Process.encoder_g_act,name="mlp_in_encoder_g"),
                                            name="encoder")
        
        self.blocks=[]
        for i in range(self.DGN_Process.block_num):
            self.block = gn.modules.GraphNetwork(
                                        edge_model_fn=lambda: self.DGN_Process.make_mlp_model_with_layernorm(self.DGN_Process.block_elayer_num,self.DGN_Process.block_elayer_size,self.DGN_Process.block_e_act,name="mlp_in_block%s_e"%(i+1)),
                                        node_model_fn=lambda: self.DGN_Process.make_mlp_model_with_layernorm(self.DGN_Process.block_nlayer_num,self.DGN_Process.block_nlayer_size,self.DGN_Process.block_n_act,name="mlp_in_block%s_n"%(i+1)),
                                        global_model_fn=lambda: self.DGN_Process.make_mlp_model_with_layernorm(self.DGN_Process.block_glayer_num,self.DGN_Process.block_glayer_size,self.DGN_Process.block_g_act,name="mlp_in_block%s_g"%(i+1)),
                                        name="block_%s"%(i+1))
                                        #reducer=blocks.unsorted_segment_max_or_zero)
            self.blocks.append(self.block)
        
        self.pooling_e = Pooling_sum_normal_params(self.DGN_Process,self.DGN_Process.pooling_e_size,self.ph_list[0],name="Pooling_sum_normal_params_e")
        self.pooling_n = Pooling_sum_normal_params(self.DGN_Process,self.DGN_Process.pooling_n_size,self.ph_list[1],name="Pooling_sum_normal_params_n")
        self.pooling_g = Pooling_sum_normal_params(self.DGN_Process,self.DGN_Process.pooling_g_size,self.ph_list[2],name="Pooling_sum_normal_params_g")
        
        self.fullconnection = self.DGN_Process.make_mlp_model_with_layernorm(self.DGN_Process.fc_num,self.DGN_Process.fc_size,self.DGN_Process.fc_act,name="mlp_out")
  
        self.outlayer = snt.Linear(1,name="out",initializers={"w": tf.glorot_uniform_initializer(),"b": tf.zeros_initializer()},regularizers={"w": tf.contrib.layers.l2_regularizer(scale=self.DGN_Process.weight_decay),"b": tf.contrib.layers.l2_regularizer(scale=self.DGN_Process.weight_decay)})
    
        self.basis=None
        self.loss=0
        self.accuracy=0
        self.outputs=[]
        self.output_pre=None
        #print("init OK")
    

    def _build(self,input):
        
        self.outputs.append(input)#1
        #latent = input
        latent = self.encoder(input)
        #print("encoder ok")
        latent0 = latent
        self.outputs.append(latent0)#2
        self.layers.append(self.encoder)
        #print("latent0 ph:", latent0)
        for i in range(self.DGN_Process.block_num):
            #block_input = utils_tf.concat([latent0,latent],axis=1)
            block_input=latent
            latent = self.blocks[i](block_input)
            self.outputs.append(latent)#3~(3+self.DGN_Process.block_num)
            #print("latent ph:", latent)
            self.layers.append(self.blocks[i])
        
        
        pooling_e=self.pooling_e(latent[EDGES])
        self.layers.append(self.pooling_e)
        pooling_n=self.pooling_n(latent[NODES])
        self.layers.append(self.pooling_n)
        pooling_g=self.pooling_g(latent[GLOBAL])
        self.layers.append(self.pooling_g)
        
        
        #print("pooling_e:",pooling_e)
        #print("pooling_n:",pooling_n)
        #print("pooling_g:",pooling_g)

        fc_input=tf.concat([pooling_e,pooling_n,pooling_g],1)
        #fc_input=latent[GLOBAL]
        self.outputs.append(fc_input)#(3+self.DGN_Process.block_num)+1
        #print("fc_input:",fc_input)
        fc_out=self.fullconnection(fc_input)
        self.outputs.append(fc_out)#(3+self.DGN_Process.block_num)+2
        self.layers.append(self.fullconnection)
        #print("fc_out:",fc_out)
        self.basis=fc_out
        
        output=self.outlayer(fc_out)
        self.outputs.append(output)#(3+self.DGN_Process.block_num)+3
        self.layers.append(self.outlayer)
        
        self.output_pre=output
        #print("output:",output)

        return self.output_pre
            
    def get_loss(self):
        
        self.loss=0
        graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_regularization_loss = tf.reduce_sum(graph_regularizers)
        self.loss+=total_regularization_loss
        self.loss+=tf.reduce_mean(tf.square(self.output_pre - self.DGN_Process.target_ph))

        return self.loss

    def get_accuracy(self):
        self.accuracy=tf.reduce_mean(tf.square(self.output_pre - self.DGN_Process.target_ph))
        return self.accuracy



class Pooling_sum_normal_params(snt.AbstractModule):
    def __init__(self,DGN_Process,output_dim,transition_matrix,name="Pooling_sum_normal_params"):
        super(Pooling_sum_normal_params, self).__init__(name=name)
        self.output_dim=output_dim
        self.transition_matrix=transition_matrix
        self.DGN_Process=DGN_Process
        self.lin_x_2_h = snt.Linear(output_size=self.output_dim,name="LinearIn%s"%name,use_bias=False,initializers={"w": tf.glorot_uniform_initializer()},regularizers={"w": tf.contrib.layers.l2_regularizer(scale=self.DGN_Process.weight_decay)})
    
    
    
    def _build(self, inputs):
        #print("lin_x_2_h:",lin_x_2_h)
        #out=lin_x_2_h(inputs)
        #out=tf.nn.softmax(inputs)
        out=tf.matmul(self.transition_matrix,tf.nn.softmax(self.lin_x_2_h(inputs)))
        #print("out in pooling: ",out)
        return out
