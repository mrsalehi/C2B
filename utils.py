import cv2
import numpy as np
import glob
import imageio
from torchvision.io import VideoReader, read_image
from tqdm import tqdm
import torch


def img_seq_gen(root, extension, S, batch_size):
    files = sorted(glob.glob(f'{root}/*.{extension}'))
    
    for i in range(0, len(files), S * batch_size):
        imgs = torch.stack([read_image(file) for file in files[i:i+S*batch_size]])
        if len(imgs) % S != 0:
            break
        yield imgs.view(len(imgs)//S, S, *imgs.shape[1:])


def video_gen(video_path, S):
    reader = VideoReader(video_path)
    for frame in reader:
        yield frame['data']


def get_simple_scheme(nbhd_height, nbhd_width):
    scheme = torch.eye(nbhd_height * nbhd_width).float()
    return scheme.view(nbhd_height * nbhd_width, nbhd_height, nbhd_width)


def load_img_seq(path, extension: str):
    img_files = glob.glob(path + '/*.jpg')
    
    imgs = []
    
    with tqdm(total=len(img_files)) as pbar:
        for file in img_files:
            imgs.append(imageio.imread(file))
            pbar.update(1)
            
    imgs = np.array(imgs) / 255.
    
    return imgs


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
    
        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame / 255.)

    # When everything done, release the capture
    cap.release()
    return np.array(frames)



if __name__ == "__main__":
    # path = '/scratch/ondemand23/mrsalehi/original_high_fps_videos/720p_240fps_1.mov'
    # subframes = load_video(path)
    # gen = img_seq_gen('videos/airboard_1/airboard_1/240/airboard_1', 'jpg', 8)
    # out = next(gen)
    # print(out.shape)
    # for img in img_gen('videos/airboard_1/airboard_1/240/airboard_1/00001.jpg'):
    #     print(img.shape)
        
    for frame in video_gen('videos/video1.mp4'):
        print(frame.shape)