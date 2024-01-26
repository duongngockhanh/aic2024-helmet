import cv2
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_video', '-sv', type=int, default=1, help="visualize from start video")
parser.add_argument('--end_video', '-ev', type=int, default=3, help="visualize to end video")
parser.add_argument('--only_video', '-ov', type=int, default=None, help="visualize a specific video")
opt = parser.parse_args()
start_video = opt.start_video
end_video = opt.end_video + 1
only_video = opt.only_video

# Read txt file as a DataFrame
data_train_dir = "aicity2024_track5_train"
file_name = "gt.txt"
file_path = os.path.join(data_train_dir, file_name)
columns = ['video_id', 'frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'class']
df = pd.read_csv(file_path, header=None, names=columns)

# Add 4 columns
df['bb_right'] = df['bb_left'] + df['bb_width']
df['bb_bot'] = df['bb_top'] + df['bb_height']
df['bb_center_x'] = df['bb_left'] + df['bb_width'] / 2
df['bb_center_y'] = df['bb_top'] + df['bb_height'] / 2

# Re-order DataFrame 
columns_order = ['video_id', 'frame', 'bb_left', 'bb_top', 'bb_right', 'bb_bot', 'bb_center_x', 'bb_center_y', 'bb_width', 'bb_height', 'class']
df = df[columns_order]


# Visualize video 001.mp4
if only_video != None:
    start_video = only_video
    end_video = only_video + 1

for i in range(start_video, end_video, 1):
    video_id = i
    video_name = f"{video_id:03d}.mp4"
    videos_dir = "videos"
    video_path = os.path.join(data_train_dir, videos_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frame = 1

    print('Press "q" to skip this video ...')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            objs = df.loc[(df['video_id'] == video_id) & (df['frame'] == num_frame),['bb_left', 'bb_top', 'bb_right', 'bb_bot']].values.tolist()
            num_frame += 1
            for obj in objs:
                x1y1 = (obj[0], obj[1])
                x2y2 = (obj[2], obj[3])
                cv2.rectangle(frame, x1y1, x2y2, color=(0, 255, 0), thickness=2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(fps) & 0xff == ord('q'):
                break
        else:
            break

print('End.')
cap.release()
cv2.destroyAllWindows()