from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from numpy import genfromtxt
from numpy import concatenate
from numpy import savetxt
import numpy as np

import json
import random
import time
import io

import boto3
from boto3.s3.transfer import TransferConfig

# s3_client = boto3.client('s3')
# bucket_name = 'serverless-bound'
# config = TransferConfig(use_threads=False)
# filename = "/tmp/Digits_Test.txt"
# f = open(filename, "wb")
# s3_client.download_fileobj(bucket_name, "ML_Pipeline/Digits_Test_Small.txt" , f, Config=config)
# f.close()
# print("Init download ###########################")

def lambda_handler(event, context):
    if('dummy' in event) and (event['dummy'] == 1):
        print("Dummy call, doing nothing")
        return
    s3_client = boto3.client('s3')
    bucket_name = 'serverless-bound'
    config = TransferConfig(use_threads=False)

    start_time = int(round(time.time() * 1000))

    start_download = int(round(time.time() * 1000))
    filename = "/tmp/Digits_Train.txt"
    f = open(filename, "wb")
    s3_client.download_fileobj(bucket_name, "ML_Pipeline/Digits_Train.txt" , f, Config=config)
    f.close()
    end_download = int(round(time.time() * 1000))

    start_process = int(round(time.time() * 1000))
    #filename = "/tmp/Digits_Test.txt"
    #f = open(filename, "wb")
    #s3_client.download_fileobj(bucket_name, "LightGBM_Data_Input/Digits_Test_Small.txt" , f, Config=config)
    #f.close()

    train_data = genfromtxt('/tmp/Digits_Train.txt', delimiter='\t')
    #test_data = genfromtxt('/tmp/Digits_Test.txt', delimiter='\t')

    train_labels = train_data[:,0]
    #test_labels = test_data[:,0]

    A = train_data[:,1:train_data.shape[1]]
    #B = test_data[:,1:test_data.shape[1]]

    # calculate the mean of each column
    MA = mean(A.T, axis=1)
    #MB = mean(B.T, axis=1)

    # center columns by subtracting column means
    CA = A - MA
    #CB = B - MB

    # calculate covariance matrix of centered matrix
    VA = cov(CA.T)

    # eigendecomposition of covariance matrix
    values, vectors = eig(VA)

    # project data
    PA = vectors.T.dot(CA.T)
    #PB = vectors.T.dot(CB.T)

    np.save("/tmp/vectors_pca.txt", vectors)

    #savetxt("/tmp/vectors_pca.txt", vectors, delimiter="\t")
    #vectors.tofile("/tmp/vectors_pca.txt")

    #print("vectors shape:")
    #print(vectors.shape)


    first_n_A = PA.T[:,0:100].real
    #first_n_B = PB.T[:,0:10].real
    train_labels =  train_labels.reshape(train_labels.shape[0],1)
    #test_labels = test_labels.reshape(test_labels.shape[0],1)

    first_n_A_label = concatenate((train_labels, first_n_A), axis=1)
    #first_n_B_label = concatenate((test_labels, first_n_B), axis=1)

    savetxt("/tmp/Digits_Train_Transform.txt", first_n_A_label, delimiter="\t")
    #savetxt("/tmp/Digits_Test_Transform.txt", first_n_B_label, delimiter="\t")

    end_process = int(round(time.time() * 1000))

    start_upload = int(round(time.time() * 1000))
    s3_client.upload_file("/tmp/vectors_pca.txt.npy", bucket_name, "ML_Pipeline/vectors_pca.txt", Config=config)
    s3_client.upload_file("/tmp/Digits_Train_Transform.txt", bucket_name, "ML_Pipeline/train_pca_transform_2.txt", Config=config)

    #s3_client.upload_file("/tmp/Digits_Test_Transform.txt", bucket_name, "LightGBM_Data/test_pca_transform.txt", Config=config)

    end_upload = int(round(time.time() * 1000)) 
    end_time = int(round(time.time() * 1000))

    res = [end_download - start_download, end_process - start_process, end_upload - start_upload, end_time - start_time]
    res = [i/1000 for i in res]
    
    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }

if __name__ == '__main__':
    res = lambda_handler({'dummy': 0}, None)
    print(res)