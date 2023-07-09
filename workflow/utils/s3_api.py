import boto3

s3_bucket_default = 'serverless-bound'

def get_dir_size(dir: str):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(
                Bucket=s3_bucket_default,
                Prefix=dir
            )
    sizes = []
    for obj in response['Contents']:
        key = obj['Key']
        size = obj['Size']
        sizes.append(size)
        
    return sum(sizes)
        
if __name__ == '__main__':
    res = get_dir_size('tpcds/dsq95/stage0/intermediate')
    print(res)