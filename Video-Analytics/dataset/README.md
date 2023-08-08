# Download the Videos

Before running the video analytics application, you are supposed to crawl the video from Youtube and upload the files in **./data/clip_data/** to remote storages (e.g., S3).

Run the following commands to crawl the videos in *_video_list.txt.
```
sudo apt install ffmpeg
pip install pytube
pip install moviepy
mkdir -p ./data/clip_data/

```
If you want to generate a new **video_list** (the provided url may expire), please replace the **API_KEY** with your own in [video_list.py](./video_list.py) and run the command.
```
python video_list.py
```