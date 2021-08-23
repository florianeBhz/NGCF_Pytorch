from utility.helper import recall_at_k
from utility.helper import ndcg_at_k
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq
import torch

cores = multiprocessing.cpu_count() // 2

#args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 64
layer_size=[64,64,64]
reg = 1e-4 #1e-5, 1e−4 , 1e1 , 1e2
lr_ =  0.0005 #0.001 #{0.0001, 0.0005, 0.001, 0.005}
   
batch_size=100
dataset = "gowalla" #"amazon-book" #

node_dropout = [0.2]
mess_dropout = [0.0,0.0,0.0]

layer_size = [64,64,64]
regs = [1e-5]
Ks = [20, 40, 60, 80, 100]
verbose = 1

args = {'device':device,'verbose':verbose,'adj_type':'single', #symetric
           'batch_size':batch_size, 'embed_size':embed_size,
           'node_dropout':node_dropout, 'mess_dropout':mess_dropout,
           'layer_size':layer_size, 'regs':regs, 'data_path':'../Data/',
            'dataset':dataset,'epoch':500,'Ks':[20, 40, 60, 80, 100], 'lr':lr_,
           'save_flag':1,'weights_path':'../orig_weights/','results_path':'../orig_results/'}

    
data_generator = Data(path=args['data_path'] + args['dataset'], batch_size=args['batch_size'])
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args['batch_size']



def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    relevance_binary = []
    for i in K_max_item_score:
        if i in user_pos_test:
            relevance_binary.append(1)
        else:
            relevance_binary.append(0)
    return relevance_binary


def get_performance(user_pos_test, relevance_binary, Ks):
    recall, ndcg = [], []

    for K in Ks:
        recall.append(recall_at_k(relevance_binary, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(relevance_binary, K, user_pos_test))

    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}



def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))
    
    relevance_binary = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, relevance_binary, Ks)



def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=False)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=False)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=True)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
           

    assert count == n_test_users
    pool.close()
    return result
