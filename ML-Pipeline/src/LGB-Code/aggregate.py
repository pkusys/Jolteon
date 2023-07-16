import boto3
import numpy as np
import lightgbm as lgb
from boto3.s3.transfer import TransferConfig
import os
import time
from utils import MergedLGBMClassifier, get_files

bucket_name = 'serverless-bound'

def aggregate_models(num_tasks, task_id, key):
    start = int(round(time.time() * 1000)) / 1000.0
    
    start_download = int(round(time.time() * 1000)) / 1000.0
    
    # Read test data from S3
    data_file = "/tmp/Digits_Test_Transform.txt"
    s3_client = boto3.client('s3')
    config = TransferConfig(use_threads=False)
    
    f = open(data_file, "wb")
    s3_client.download_fileobj(bucket_name, "ML_Pipeline/train_pca_transform_2.txt" , f, Config=config)
    f.close()
    test_data = np.genfromtxt(data_file, delimiter='\t')
    Y_test = test_data[5000:,0]
    X_test = test_data[5000:,1:test_data.shape[1]]
    
    # Read models from S3
    files = get_files(bucket_name, key)
    number = len(files)
    
    remain = number % num_tasks
    number = number // num_tasks
    
    if task_id < remain:
        start_id = task_id * (number + 1)
        end_id = start_id + number + 1
    else:
        start_id = remain * (number + 1) + (task_id - remain) * number
        end_id = start_id + number
    
    process_files = files[start_id:end_id]
    model_list = []
    
    for i in range(len(process_files)):
        s3_client = boto3.client('s3')
        config = TransferConfig(use_threads=False)
        filename = "/tmp/lightGBM_model_{}.txt".format(i)
        f = open(filename, "wb")
        s3_client.download_fileobj(bucket_name, process_files[i], f, Config=config)
        f.close()
        
        model = lgb.Booster(model_file=filename)
        model_list.append(model)
        
        os.remove(filename)
        
    end_download = int(round(time.time() * 1000)) / 1000.0
    
    start_process = int(round(time.time() * 1000)) / 1000.0
    
    # Merge models
    forest = MergedLGBMClassifier(model_list)
    model_fn = "/tmp/Forest_model.txt"
    s3_model_key = "ML_Pipeline/stage2/Forest_model_{}.txt".format(task_id)
    forest.save_model(model_fn)
    
    # Predict
    y_pred = forest.predict(X_test)
    count_match=0
    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == Y_test[i]:
            count_match = count_match + 1

    # The accuracy on the training set  
    acc = count_match / len(y_pred)
    pred_fn = "/tmp/Predict.txt"
    s3_pred_key = "ML_Pipeline/stage2/Predict_{}.txt".format(task_id)
    np.savetxt(pred_fn, y_pred, delimiter='\t')
    
    end_process = int(round(time.time() * 1000)) / 1000.0
    
    start_upload = int(round(time.time() * 1000)) / 1000.0
    s3_client.upload_file(model_fn, bucket_name, s3_model_key, Config=config)
    s3_client.upload_file(pred_fn, bucket_name, s3_pred_key, Config=config)
    end_upload = int(round(time.time() * 1000)) / 1000.0
    
    end = int(round(time.time() * 1000)) / 1000.0
    
    return [end_download - start_download, end_process - start_process, end_upload - start_upload, end - start, acc]
    
    
    
    
if __name__ == '__main__':
    key = 'ML_Pipeline/stage1/lightGBM_model'
    
    res = aggregate_models(2, 0, key)
    
    print(res)
    
    res = aggregate_models(2, 1, key)
    
    print(res)