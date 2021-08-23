import torch
import torch.optim as optim

from NGCF import MyNGCF, NGCFConv
from utility.helper import *
from utility.batch_test import *

import torch
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')
from time import time


if __name__ == '__main__':
    
    
    A, L_I, L = data_generator.get_adj_mat(args['adj_type']) 
    
    N,M = data_generator.n_users, data_generator.n_items
    model = MyNGCF(N,M,L,L_I,device,embed_size,reg,layer_size,batch_size,node_dropout[0],mess_dropout[0]).to(device)
    
    #Training--------------------------------------------------------------------------------
    t0 = time()
    
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    loss_loger, rec_loger, ndcg_loger = [], [], []
    for epoch in range(args['epoch']):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args['batch_size'] + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,drop_flag=1)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args['verbose'] > 0 and epoch % args['verbose'] == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        ndcg_loger.append(ret['ndcg'])

        if args['verbose'] > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                        'ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=50)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args['save_flag'] == 1:
            torch.save(model.state_dict(), args['weights_path'] + str(epoch) + '.pkl')
            print('save the weights in path: ', args['weights_path'] + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
   
     
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    save_path = '%soutput/%s/%s.result' % (args['results_path'] % args['dataset'])
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args['embed_size'], 0.0001, args['layer_size'], args['node_dropout'], args['mess_dropout'], args['regs'],
           args['adj_type'], final_perf))
    
    f.close()