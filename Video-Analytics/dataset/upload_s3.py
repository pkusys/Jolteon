import subprocess
import os

bucketName = 'serverless-bound'

def upload_to_s3(li : list):
    fns = []
    
    for idx in li:
        assert isinstance(idx, int)
        fn = './data/clip_data/video_' + str(idx) + '.mp4'
        fns.append(fn)
    
    for fn in fns:
        assert isinstance(fn, str)
        assert os.path.exists(fn)
        cmd = ['aws', 's3', 'cp', fn, 's3://{}/Video-Analytics/dataset/'.format(bucketName)]
        subprocess.run(cmd)
        
if __name__ == '__main__':
    upload_to_s3([0, 1, 2, 3])
    