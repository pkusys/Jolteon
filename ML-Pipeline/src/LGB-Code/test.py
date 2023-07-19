import boto3
import numpy as np
from boto3.s3.transfer import TransferConfig
import time
from utils import get_files

bucket_name = 'serverless-bound'

def test_forests(key):
    start = int(round(time.time() * 1000)) / 1000.0
    
    start_download = int(round(time.time() * 1000)) / 1000.0
    input_address = key['input_address']
    assert len(input_address) == 2
    
    files = get_files(bucket_name, input_address[1])
    preds = []
    
    # Read predictions from S3
    for i in range(len(files)):
        s3_client = boto3.client('s3')
        config = TransferConfig(use_threads=False)
        filename = "/tmp/Pred_{}.txt".format(i)
        f = open(filename, "wb")
        s3_client.download_fileobj(bucket_name, files[i], f, Config=config)
        f.close()
        
        preds.append(np.genfromtxt(filename, delimiter='\t'))
        
    # Read labels from S3
    data_file = "/tmp/Digits_Test_Transform.txt"
    s3_client = boto3.client('s3')
    config = TransferConfig(use_threads=False)
    
    f = open(data_file, "wb")
    s3_client.download_fileobj(bucket_name, input_address[0], f, Config=config)
    f.close()
    test_data = np.genfromtxt(data_file, delimiter='\t')
    Y_test = test_data[5000:,0]
    
    end_download = int(round(time.time() * 1000)) / 1000.0
    
    start_process = int(round(time.time() * 1000)) / 1000.0
    y_pred = sum(preds) / len(preds)
    count_match=0
    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == Y_test[i]:
            count_match = count_match + 1
    acc = count_match / len(y_pred)
    
    end_process = int(round(time.time() * 1000)) / 1000.0
    
    end = int(round(time.time() * 1000)) / 1000.0
    
    return [end_download - start_download, end_process - start_process, 0, end - start, acc]


if __name__ == '__main__':
    key = "ML_Pipeline/stage2/Predict"
    print(test_forests(key))