import lightgbm as lgb
import boto3
import numpy as np
from boto3.s3.transfer import TransferConfig
import random
import time
from multiprocessing import Process, Manager
from utils import MyPool


def train_with_multprocess(task_id, num_process, num_vcpu, loc):
    start = int(round(time.time() * 1000)) / 1000.0
    ia = loc['input_address'][0]
    oa = loc['output_address'][0]
    processes = []
    manager = Manager()
    res_dict = manager.dict()
    param = {
        'feature_fraction': 1,
        'max_depth': 8,
        'num_of_trees': 30,
        'chance': 1
    }

    for i in range(num_process):
        feature_fraction = round(random.random()/2 + 0.5, 1)
        chance = round(random.random()/2 + 0.5, 1)
        
        param['feature_fraction'] = feature_fraction
        param['chance'] = chance
        
        pro = Process(target=train, args=(task_id, i, param['feature_fraction'],\
                    param['max_depth'], param['num_of_trees'], param['chance'], ia, oa, res_dict))
        processes.append(pro)
    
    start_ids = 0
    end_ids = start_ids + num_vcpu
    
    while start_ids < num_process:
        end_ids = min(end_ids, num_process)
        for i in range(start_ids, end_ids):
            processes[i].start()
            
        for i in range(start_ids, end_ids):
            processes[i].join()
        
        start_ids += num_vcpu
        end_ids += num_vcpu
    
    # pool = MyPool(num_vcpu, processes)
    # pool.run()
        
    end = int(round(time.time() * 1000)) / 1000.0
    
    avg_acc = 0
    avg_read = 0
    avg_compute = 0
    avg_write = 0
    
    for i in res_dict:
        avg_acc += res_dict[i][0]
        avg_read += res_dict[i][1]
        avg_compute += res_dict[i][2]
        avg_write += res_dict[i][3]
        
    avg_acc = avg_acc / len(res_dict)
    avg_read = avg_read / num_vcpu
    avg_compute = avg_compute / num_vcpu
    avg_write = avg_write / num_vcpu
    
    return [avg_read, avg_compute, avg_write, end - start, avg_acc]

def train(task_id, process_id, feature_fraction, max_depth, num_of_trees, chance, ia, oa, res_dict):
    assert isinstance(process_id, int)
    assert isinstance(task_id, int)
    start = int(round(time.time() * 1000)) / 1000.0
    
    # print(feature_fraction, max_depth, num_of_trees, chance)
    
    s3_client = boto3.client('s3')
    config = TransferConfig(use_threads=False)
    filename = "/tmp/Digits_Train_Transform_{}.txt".format(process_id)
    bucket_name = 'serverless-bound'
    
    f = open(filename, "wb")
    start_download = int(round(time.time() * 1000)) / 1000.0
    s3_client.download_fileobj(bucket_name, ia, f, Config=config)
    end_download = int(round(time.time() * 1000)) / 1000.0
    f.close()
    
    train_data = np.genfromtxt(filename, delimiter='\t')
    Y_train = train_data[0:5000,0]
    X_train = train_data[0:5000,1:train_data.shape[1]]
    
    _id=str(task_id) + "_" + str(process_id)
    # chance = round(random.random()/2 + 0.5,1)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_classes' : 10,
        'metric': {'multi_logloss'},
        'num_leaves': 50,
        'learning_rate': 0.05,
        'feature_fraction': feature_fraction,
        'bagging_fraction': chance, # If model indexes are 1->20, this makes feature_fraction: 0.7->0.9
        'bagging_freq': 5,
        'max_depth': max_depth,
        'verbose': -1,
        'num_threads': 2
    }
    
    start_process = int(round(time.time() * 1000)) / 1000.0
    lgb_train = lgb.Dataset(X_train, Y_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_of_trees,
                    valid_sets=lgb_train,
                    # early_stopping_rounds=5
                    )
    
    y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    count_match=0
    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == Y_train[i]:
            count_match = count_match + 1
    # The accuracy on the training set  
    acc = count_match/len(y_pred)
    end_process = int(round(time.time() * 1000)) / 1000.0
    
    model_name = '_tree_' + str(_id)
    gbm.save_model("/tmp/" + model_name)
    start_upload = int(round(time.time() * 1000)) / 1000.0
    s3_client.upload_file("/tmp/" + model_name, bucket_name, oa + model_name, Config=config)
    end_upload = int(round(time.time() * 1000)) / 1000.0
    
    end = int(round(time.time() * 1000)) / 1000.0
    
    res_dict[process_id] = [acc, end_download-start_download, end_process-start_process, \
            end_upload-start_upload, end - start]
    
    return None
    

if __name__ == '__main__':
    loc = {}
    loc['input_address'] = ['ML_Pipeline/stage0/train_pca_transform.txt']
    loc['output_address'] = ['ML_Pipeline/stage1/model']
    t0 = time.time()
    res = train_with_multprocess(0, 2, 1, loc)
    t1 = time.time()
    
    print("Total time: ", t1-t0)