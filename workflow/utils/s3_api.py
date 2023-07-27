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

def clear_data(dir: str):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(s3_bucket_default)

    objects_to_delete = []
    for obj in bucket.objects.filter(Prefix=dir):
        objects_to_delete.append({'Key': obj.key})
    
    if len(objects_to_delete) > 0:
        response = bucket.delete_objects(Delete={'Objects': objects_to_delete})
        if 'Errors' in response:
            raise Exception(response['Errors'])
        
if __name__ == '__main__':
    # res = get_dir_size('tpcds/dsq95/stage0/intermediate')
    clear_data('tpcds/dsq95/stage')