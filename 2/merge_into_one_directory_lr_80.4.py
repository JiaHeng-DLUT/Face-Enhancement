import os
import pandas as pd
import shutil
df = pd.read_csv('/home/jiaheng/Desktop/GitHub/IJB-C/FaceQnet/lr_80.4.txt', sep=' ', header=None)
df = df.rename({0:"idx", 1:"video_id", 2:"filename"}, axis="columns")
print(df.head(3))
print(df.shape[0])
print(len(pd.unique(df['filename'])))

base_dir = '/home/jiaheng/Desktop/GitHub/IJB-C-1/LR'
target_dir = '/home/jiaheng/Desktop/LR_804_in_one_dir'

def process_row(s):
    idx = s['idx']
    video_id = s['video_id']
    filename = s['filename']
    img_dir = base_dir + '/' + str(idx) + '/' + str(video_id)
    img_path = img_dir + '/' + str(filename)
    target_path = target_dir + '/' + str(filename)
    shutil.copyfile(img_path, target_path)
    print(img_dir)

os.makedirs(target_dir, exist_ok=True)
# df = df.head(3)
df.apply(process_row, axis=1)

