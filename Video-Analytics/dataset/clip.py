from moviepy.video.io.VideoFileClip import VideoFileClip
import subprocess
import os
import shutil

def clip_video(video_path, start_time, end_time, output_path):
    video_clip = VideoFileClip(video_path)
    clipped_video = video_clip.subclip(start_time, end_time)
    clipped_video = clipped_video.set_fps(0.5)
    clipped_video.write_videofile(output_path)
    
def adjust_video_bitrate(input_path, output_path, target_bitrate = '400k'):
    # use ffmpeg 
    command = [
        'ffmpeg',
        '-i', input_path,
        '-b:v', target_bitrate,
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(command)
    
if __name__ == '__main__':
    dir_path = './data/'
    out_path = './data/clip_data'
    number = 16
    
    file_names = os.listdir(dir_path)
    cnt = 0

    for file_name in file_names:
        if file_name.startswith('raw_video') and file_name.endswith('.mp4'):
            idx = file_name.removeprefix('raw_video_')
            idx = idx.removesuffix('.mp4')
            output_path = os.path.join(out_path, 'video_{}.mp4'.format(idx))
            video_path = os.path.join(dir_path, file_name)
            clip_video(video_path, 0, 60, output_path)
            cnt += 1   
    
    if cnt == number:
        print('Successfully clip {} videos'.format(cnt))
        
    # file_names = os.listdir(out_path)
    # cnt = 0
    
    # for file_name in file_names:
    #     if file_name.startswith('video') and file_name.endswith('.mp4'):
    #         tmp_fn = '/tmp/tmp_video.mp4'
    #         video_path = os.path.join(out_path, file_name)
    #         adjust_video_bitrate(video_path, tmp_fn)
    #         os.remove(video_path)
    #         shutil.move(tmp_fn, video_path)
    #         cnt += 1
    
    # if cnt == number:
    #     print('Successfully adjust code rate for {} videos'.format(cnt))