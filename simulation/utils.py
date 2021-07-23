import cv2
import numpy as np


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


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
    return np.stack(frames)



if __name__ == "__main__":
    path = '/scratch/ondemand23/mrsalehi/original_high_fps_videos/720p_240fps_1.mov'
    subframes = load_video(path)
