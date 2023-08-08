from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def clip_video(video_path, start_time, end_time, output_path):
    video_clip = VideoFileClip(video_path)
    clipped_video = video_clip.subclip(start_time, end_time)
    clipped_video.write_videofile(output_path)
    
if __name__ == '__main__':
    dir_path = './data/'
    out_path = './data/clip_data'
    number = 100
    
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