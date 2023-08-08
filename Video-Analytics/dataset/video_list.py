import requests

API_KEY = 'Your API Key'
max_results = 200
results_per_page = 50

# time unit is second
def get_info(video_id) -> int:
    base_url = 'https://www.googleapis.com/youtube/v3/videos'
    params = {
        'key': API_KEY,
        'id': video_id,
        'part': 'contentDetails,status',
    }

    response = requests.get(base_url, params=params)
    video_info = response.json()['items'][0]
    
    duration_iso8601 = video_info['contentDetails']['duration']
    duration_seconds = 0
    video_details = video_info['contentDetails']
    if 'contentRating' in video_details and 'ytAgeRestricted' in video_details['contentRating']:
        is_age_restricted =  True
    else:
        is_age_restricted = False

    if duration_iso8601.startswith('PT'):
        duration_iso8601 = duration_iso8601[2:]
        part = ''
        hours = 0
        minutes = 0
        seconds = 0
        for char in duration_iso8601:
            if char.isdigit():
                part += char
            else:
                if char == 'H':
                    hours = int(part)
                elif char == 'M':
                    minutes = int(part)
                elif char == 'S':
                    seconds = int(part)
                part = ''
        duration_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError('Invalid duration string')
    
    return duration_seconds, is_age_restricted

def get_list(tag):
    total_results =[]
    base_url = 'https://www.googleapis.com/youtube/v3/search'
    page_token = None
    while len(total_results) < max_results:
        params = {
            'key': API_KEY,
            'q': tag,
            'maxResults': results_per_page,
            'type': 'video',
            'eventType': 'completed',  # 排除正在直播的视频
            'videoDuration': 'short',
            'pageToken': page_token,
        }

        response = requests.get(base_url, params=params)
        page_results = response.json()['items']
        total_results.extend(page_results)

        if 'nextPageToken' in response.json():
            page_token = response.json()['nextPageToken']
        else:
            break
    ret = []
    
    for item in total_results:
        ret.append(item['id']['videoId'])
        
    return ret

if __name__ == '__main__':
    tag = 'MUSIC'
    # tag = 'NEWS'
    number = 100
    li = get_list(tag)
    print(len(li))
    vid_list = []
    for v_id in li:
        duration, age_dist = get_info(v_id)
        
        # limite the video size
        if duration > 61 and duration < 500 and not age_dist:
            vid_list.append(v_id)
        
        if len(vid_list) >= number:
            print('Successfully get {} videos ids in tag'.format(number), tag)
            break
    
    if len(vid_list) < number:
        raise ValueError('Not enough videos in tag {}, only {} videos idx'.format(tag, len(vid_list)))
        
    with open('{}_video_list.txt'.format(tag), 'w') as f:
        for v_id in vid_list:
            url = 'https://www.youtube.com/watch?v={}'.format(v_id)
            f.write(url + '\n')