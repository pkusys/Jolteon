import boto3
from boto3.s3.transfer import TransferConfig
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
from PIL import Image

def calculate_average_pixel_value(image):
    # Convert image to grayscale image
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    
    # Calculate the average value of pixels
    average_pixel_value = np.mean(gray_image)
    
    return average_pixel_value

def get_suffix(filename):
    fn = filename.split('.')[0]
    fn = fn.split('_')
    file_id = fn[-2]
    chunk_id = fn[-1]
    
    return file_id, chunk_id

bucketName = 'serverless-bound'

def extrace_frames(input_adresses, output_adress):
    assert isinstance(input_adresses, list)
    if len(input_adresses) > 0:
        assert isinstance(input_adresses[0], str)
        
    s3_client = boto3.client('s3')
    config = TransferConfig(use_threads=False)
    
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
        
        start_comp = int(round(time.time() * 1000)) / 1000.0
        file_id, chunk_id = get_suffix(s3_fn)
        best_frame = None
        best_metric = float('-inf')
        video_clip = VideoFileClip(filename, verbose=False)
        
        for frame in video_clip.iter_frames(fps=0.5, dtype='uint8'):
            frame_metric = calculate_average_pixel_value(frame)
            if frame_metric > best_metric:
                best_metric = frame_metric
                best_frame = frame
        pil_image = Image.fromarray(best_frame)
        
        end_comp = int(round(time.time() * 1000)) / 1000.0
        comp_time += end_comp - start_comp
        
        start_write = int(round(time.time() * 1000)) / 1000.0
        
        tmp_filename = "/tmp/representative_frame.jpg"
        pil_image.save(tmp_filename)
        
        out_a = output_adress + '_' + file_id + '_' + chunk_id + '.jpg'
        s3_client.upload_file(tmp_filename, bucketName, out_a, Config=config)
        
        video_clip.close()
        
        end_write = int(round(time.time() * 1000)) / 1000.0
        write_time += end_write - start_write
        
    end = int(round(time.time() * 1000)) / 1000.0
    
    return [read_time, comp_time, write_time, end - start]


if __name__ == '__main__':
    file_li = ['Video-Analytics/stage0/clip_video_0_0.mp4']
    output_adress = 'Video-Analytics/stage1/clip_frame'
    ret = extrace_frames(file_li, output_adress)
    print(ret)