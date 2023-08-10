import boto3
from boto3.s3.transfer import TransferConfig
import time
from moviepy.video.io.VideoFileClip import VideoFileClip

bucketName = 'serverless-bound'

def split_videos(input_adresses, output_adress, task_id = 0, file_start_id = 0, chunk_size = 10):
    assert isinstance(input_adresses, list)
    if len(input_adresses) > 0:
        assert isinstance(input_adresses[0], str)
    
    start = int(round(time.time() * 1000)) / 1000.0
    s3_client = boto3.client('s3')
    config = TransferConfig(use_threads=False)
    read_time = 0
    write_time = 0
    comp_time = 0
    
    for idx, s3_fn in enumerate(input_adresses):
        start_read = int(round(time.time() * 1000)) / 1000.0
        filename = "/tmp/src.mp4"
        f = open(filename, "wb")
        s3_client.download_fileobj(bucketName, s3_fn, f, Config=config)
        f.close()
        end_read = int(round(time.time() * 1000)) / 1000.0
        read_time += end_read - start_read
        
        file_id = idx + file_start_id
        
        start_comp = int(round(time.time() * 1000)) / 1000.0
        vc = VideoFileClip(filename, verbose=False)
        vc.write_videofile
        video_len = int(vc.duration)
        
        start_size = 0
        cnt = 0
        end_comp = int(round(time.time() * 1000)) / 1000.0
        comp_time += end_comp - start_comp
        
        while start_size < video_len:
            start_comp = int(round(time.time() * 1000)) / 1000.0
            end_size = min(start_size + chunk_size, video_len)
            tmp_filename = "/tmp/clip.mp4"
            
            clip_vc = vc.subclip(start_size, end_size)
            clip_vc.write_videofile(tmp_filename, temp_audiofile="/tmp/temp-audio.mp3", logger=None)
            del clip_vc
            end_comp = int(round(time.time() * 1000)) / 1000.0
            comp_time += end_comp - start_comp
            
            start_write = int(round(time.time() * 1000)) / 1000.0
            s3_clip_name = output_adress + '_' + str(file_id) +  '_' + str(cnt) + '.mp4'
            
            s3_client.upload_file(tmp_filename, bucketName, s3_clip_name, Config=config)
            
            cnt += 1
            start_size += chunk_size
            end_write = int(round(time.time() * 1000)) / 1000.0
            write_time += end_write - start_write
        
        vc.close()
    end = int(round(time.time() * 1000)) / 1000.0
    
    return [read_time, comp_time, write_time, end - start]
    
        
        
if __name__ == '__main__':
    file_li = ['Video-Analytics/dataset/video_2.mp4']
    output_adress = 'Video-Analytics/stage0/clip_video'
    ret = split_videos(file_li, output_adress)
    print(ret)