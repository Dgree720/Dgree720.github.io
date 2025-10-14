# 1Unit01_3ytb.py
import yt_dlp
import cv2

video_url = "https://www.youtube.com/watch?v=z_fY1pj1VBw"
# video_url = "https://www.youtube.com/watch?v=aIODuVh2T-Y"
# video_url = "https://www.youtube.com/watch?v=Fl1ReheFtDw"
ydl_opts = {"format": "best", "quiet": True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
stream_url = info_dict["url"]

cap = cv2.VideoCapture(stream_url)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Unit01_3 | 322022 | ", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 'esc' 退出
        break
cap.release()
cv2.destroyAllWindows()
