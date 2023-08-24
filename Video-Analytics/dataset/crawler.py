from pytube import YouTube
import pytube

def download_video(video_url, output_path, file_id):
    try:
        yt = YouTube(video_url)
        streams = yt.streams.filter(progressive=True, file_extension='mp4')
        chosen_stream = streams.get_by_resolution('720p')
        if chosen_stream is None:
            return 2
        chosen_stream.download(output_path=output_path, filename='raw_video_{}.mp4'.format(file_id))
        return 0
    except Exception as e:
        print(e)
        if isinstance(e, pytube.exceptions.AgeRestrictedError):
            return 2
        return 1

if __name__ == '__main__':
    tags = ['MUSIC', 'NEWS']
    numbers = [16, 16]
    cnt = 0
    for tag, num in zip(tags, numbers):
        local_cnt = 0
        with open('{}_video_list.txt'.format(tag), 'r') as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines]
        for url in lines:
            while True:
                ret = download_video(url, './data/', cnt)
                if ret == 0:
                    cnt += 1
                    local_cnt += 1
                    break
                # restrict video or other error due to permission
                elif ret == 2:
                    break
            if local_cnt >= num:
                print('Successfully get {} videos ids in tag'.format(num), tag)
                break
        
    if cnt != sum(numbers):
        raise ValueError('Not enough videos in tag {}, only {} videos idx'.format(tag, cnt))