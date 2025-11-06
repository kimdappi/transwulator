## download_mp4.py
## 감정 추출 전 영상 다운로드
import yt_dlp
from config import YOUTUBE_PATH 

YOUTUBE_URL = YOUTUBE_PATH

def download_video(url, out_template="downloaded_video.%(ext)s"):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # 영상+음성 합쳐서 MP4
        'merge_output_format': 'mp4',
        'outtmpl': out_template,
        'quiet': False,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

if __name__ == "__main__":
    video_path = download_video(YOUTUBE_URL)
    print(f"✅ 다운로드 완료: {video_path}")

## 이걸 이렇게 돌려도 돌아가는지 매우 의문이다...
!ffmpeg -i downloaded_video.mp4 -vf "fps=25" -c:v libx264 -preset fast -crf 23 -c:a copy fixed_video.mp4

