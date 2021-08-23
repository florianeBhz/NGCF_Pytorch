import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import coo_matrix
from time import time
#import torch.sparse as sparse
from scipy import sparse
import scipy.sparse as sp

import random
import torch.nn.functional as F
#from configs import configs 
import numpy as np
from torch import sparse_coo_tensor



class NGCFConv(nn.Module): 
    #class of NGCF convolution layers
    def __init__(self, in_channels, out_channels,initializer):
        super(NGCFConv, self).__init__()
        self.embedding_dim = in_channels
        self.W1 = nn.Parameter(initializer(torch.empty(in_channels,out_channels)))
        self.b1 = nn.Parameter(initializer(torch.empty(1, out_channels)))
        
        self.W2 = nn.Parameter(initializer(torch.empty(in_channels,out_channels)))
        self.b2 = nn.Parameter(initializer(torch.empty(1, out_channels)))
        
        
    def forward(self, L_I,prev_embeddings,W1,W2,b1,b2):
        #propagation rule #Eq7

        L_I_E = torch.sparse.mm(L_I, prev_embeddings)
        
        left = torch.matmul(L_I_E,W1) + b1#sum_embeddings
        right = torch.mul(prev_embeddings,L_I_E) 
        right = torch.matmul(right,W2) +b2

        return left+right 

        
    
class MyNGCF(nn.Module):
    def __init__(self,N,M,L,L_I,device,embedding_dim=64,decay=0.2,layers = [64,64,64],\
                 batch_size=1024,p1=None,p2=None,training=True):

        super(MyNGCF, self).__init__()

        self.embedding_dim = embedding_dim
        self.p1 = p1 #message_dropout 
        self.p2 = p2 #node_dropout
        self.training = training
        self.M = M
        self.N = N
        self.decay = decay
        self.batch_size = batch_size
        self.layers = layers
        self.device = device
        
        coo = L_I.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        self.L_I = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        
        #embedding initialization
        ## We use the Xavier initializer [6] to initialize the model parameters 
        initializer = nn.init.xavier_uniform_
                          
        #making users and items embeddings weights trainable
        self.parameter_list = nn.ParameterDict({
            'embed_user': nn.Parameter(initializer(torch.empty(self.N,
                                                 self.embedding_dim))),
            'embed_item': nn.Parameter(initializer(torch.empty(self.M,
                                                 self.embedding_dim)))
        })

                        
        #NGCF conv layers and weights matrices initialization
        ##when the embedding size is 64 and we use 3 propagation layers of size 64 × 64
        
        #list of stacked NGCFConv layers
        self.convs = nn.ModuleList()
        
        #embedding layer + conv layers
        layers = [self.embedding_dim] +self.layers
        
        #initializing conv layers and storing their weights and bias in the parameter dictionnary to make them trainable
        for i in range(len(self.layers)):
          layer_ = NGCFConv(layers[i], layers[i+1],initializer)
          self.convs.append(layer_)

          #initialize weights for training
          W1,b1, W2, b2 = layer_.W1,layer_.b1, layer_.W2,layer_.b2
            
          self.parameter_list.update({'W1_%d'%i: W1}) 
          self.parameter_list.update({'W2_%d'%i:W2})
          self.parameter_list.update({'b1_%d'%i:b1})
          self.parameter_list.update({'b2_%d'%i:b2})


    def forward(self,sampled_users,observed_items_idx, unobserved_items_idx,drop_flag=True):
        
        #Laplacian + Identity
        self.L_I = self.sparse_dropout(self.L_I,
                                    1-self.p2,
                                    self.L_I._nnz()) if drop_flag else self.L_I

        #initial embeddings (concatenating items and users embeddings)
        x = torch.cat([self.parameter_list['embed_user'], self.parameter_list['embed_item']], dim=0)
        final_reprs = x.clone()

        #propagation for each layer
        for i in range(len(self.layers)):
            x = self.convs[i](self.L_I,x,self.parameter_list['W1_%d'%i],self.parameter_list['W2_%d'%i],\
                              self.parameter_list['b1_%d'%i],self.parameter_list['b2_%d'%i])
            x = F.leaky_relu(x,negative_slope=0.2)
            #message dropout
            if self.p2:
              x = F.dropout(x, self.p1, self.training)
              x = F.normalize(x,dim=1,p=2)
            final_reprs = torch.cat([final_reprs,x.clone()],dim=1)

        #separate users and items
        users_,items_ = final_reprs[:self.N], final_reprs[self.N:]
        #retrieve only batched users
        sampled_users_repr = users_[sampled_users,:]
        #get observed items representations
        observed_items_repr = items_[observed_items_idx,:]
        #get unobserved items representations
        unobserved_items_repr = items_[unobserved_items_idx,:]
        self.final_reprs = final_reprs
        return sampled_users_repr, observed_items_repr,unobserved_items_repr
    
    def sparse_dropout(self, x, keep_prob, _nonzero_elems):
        random_tensor = keep_prob
        random_tensor += torch.rand(_nonzero_elems).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        pre_out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return pre_out * (1. / keep_prob)
    
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
    
    #dot product for prediction
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())