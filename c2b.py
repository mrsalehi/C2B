import torch
import cv2
import PIL
from typing import List
from utils import load_video
from time import time
from tqdm import tqdm
from imageio import imwrite
from utils import get_simple_scheme
import os
import cv2
import numpy as np


class C2BCamera:
    def __init__(self, args, config, subframe_gen):
        self.frame_height, self.frame_width = config.camera.frame_height, config.camera.frame_width
        self.nbhd_height, self.nbhd_width = config.camera.nbhd_size
        n_pixels = self.nbhd_height * self.nbhd_width
        self.S = config.camera.S
        self.device = args.device
        self.max_intensity = torch.tensor(2 ** config.camera.pixel_bit_depth)
        
        assert self.frame_height % self.nbhd_height == 0
        assert self.frame_width % self.nbhd_width == 0
        assert self.S % n_pixels == 0
        
        self.subframe_gen = subframe_gen
        
        # self.scheme = torch.FloatTensor(config.camera.scheme)
        self.scheme = get_simple_scheme(self.nbhd_height, self.nbhd_width).to(self.device)
        # scheme = torch.tile(self.scheme, (height // self.nbhd_height, width // self.nbhd_width)).to(raw_subframes.device)
        # scheme_ = scheme_.unsqueeze(1).repeat(1, S // n_pixels, 1, 1).view(S, height, width)
        
        scheme_ = torch.tile(self.scheme, (self.frame_height // self.nbhd_height, 
                                                self.frame_width // self.nbhd_width)).to(self.device) # shape: (n_pixels, height, width)
        self.scheme_ = scheme_.unsqueeze(1).repeat(1, self.S // n_pixels, 1, 1).view(self.S, self.frame_height, self.frame_width)  # shape: (S, height, width)
        self.save_output = config.camera.save_output
        self.save_dir = config.camera.save_dir
        self.batch_size = config.data.batch_size
        
        self.noise_std = None
        if config.camera.add_noise:
            self.noise_std = config.camera.noise_std
        
        if self.save_output:
            os.makedirs(self.save_dir, exist_ok=True)
        
    def save_frame(self, frame, frame_num):
        for i in range(2):
            bucket_normalized = ((frame[i] / self.max_intensity) * 255.).to(torch.uint8)
            bucket_i_image = bucket_normalized.permute(1, 2, 0).cpu().numpy()
            imwrite(os.path.join(self.save_dir, f'{frame_num}_b{i}.jpg'), bucket_i_image)

    def multiplex(self, subframes: torch.Tensor):
        """
        Assuming that the number of subframes is divisible by the number of pixles in a neighborhood (n_pixels), bucket 1 is active in the following cases: 
        * pixel 1 of each neighborhood in the first S / n_pixels 
        * pixel 2 of each neighborhood in the second S / n_pixels
        * pixel 3 of each neighborhood in the third S / n_pixels
        ...
        * last pixel of each neighborhood in the last S / n_pixels
        
        subframes: intensities of subframes of a frame as a numpy array with shape (S, H, W)
        nbhd_size: tuple of width and height of the neighborhood (For now W must be divisible by the width of the nbhd 
        and H must be divisible by the height of the neighborhood)
        """
        # S, height, width = raw_subframes.shape

        # scheme_ = torch.tile(self.scheme, (height // self.nbhd_height, width // self.nbhd_width)).to(raw_subframes.device) # shape: (n_pixels, height, width)
        # scheme_ = scheme_.unsqueeze(1).repeat(1, S // n_pixels, 1, 1).view(S, height, width)  # shape: (S, height, width)
        
        c2b_frame_bucket0 = torch.minimum(torch.sum(subframes * self.scheme_.unsqueeze(1), dim=1), self.max_intensity.to(self.device))
        c2b_frame_bucket1 = torch.minimum(torch.sum(subframes * (1 - self.scheme_.unsqueeze(1)), dim=1), self.max_intensity.to(self.device))

        if self.noise_std is not None:
            c2b_frame_bucket0 += torch.randn_like(c2b_frame_bucket0) * self.noise_std
            c2b_frame_bucket1 += torch.randn_like(c2b_frame_bucket1) * self.noise_std
        
        return c2b_frame_bucket0, c2b_frame_bucket1


    def get_c2b_frames_from_preexisting_frames(self):
        """
        Computes the C2B frames (bucket 0 and bucket 1) from the frame's data of a conventional high 
        temporal resolution camera.
        """
        c2b_frames = []
        frame_idx = 0
        with torch.no_grad():
            for sf_batch in self.subframe_gen(self.S, self.batch_size):
                bucket_0, bucket_1 = self.multiplex(sf_batch.to(self.device))
                
                for i in range(self.batch_size):
                    frame = (bucket_0[i].cpu(), bucket_1[i].cpu())
                    frame_idx += 1
                    c2b_frames.append(frame)
                    if self.save_output:
                        self.save_frame(frame, i)
        
                print(f'Processed C2B Frame {frame_idx}.')
                
            # for multi_chnl_subframes in self.subframe_gen(self.S):
            #     i += 1
            #     bucket_0_chnls, bucket_1_chnls = [], []
            #     for chnl in range(multi_chnl_subframes.size(1)): # Do the simulation for each one of the color channels separtately
            #         bucket_0, bucket_1 = self.multiplex(multi_chnl_subframes[:, chnl, ...].to(self.device))
            #         bucket_0_chnls.append(bucket_0)
            #         bucket_1_chnls.append(bucket_1)
                
            #     bucket_0, bucket_1 = torch.stack(bucket_0_chnls), torch.stack(bucket_1_chnls)
            #     frame = (bucket_0.cpu(), bucket_1.cpu())
            #     c2b_frames.append(frame)
                
            #     if self.save_output:
            #         self.save_frame(frame, i)
                    
            #     print(f'Processed C2B Frame {i}.')
    
        return c2b_frames


if __name__ == '__main__':
    DEBUG = 1
    
    # W = torch.FloatTensor([[[1, 1, 0], [1, 1, 1]], [[0 ,0, 0], [1, 0, 1]]]).cuda()
    
    if DEBUG:
        # subframes = torch.FloatTensor([[[1, 1, 1], [1, 0, 0]], [[0 ,0, 0], [0, 0, 1]]]).cuda()
        sf = torch.FloatTensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15 ,16]])
        subframes = torch.stack([sf, -sf, sf - 100, sf - 200, sf - 300, sf - 400, sf - 500, sf - 600], dim=0).cuda() 
    else:
        path = '/scratch/ondemand23/mrsalehi/original_high_fps_videos/720p_240fps_1.mov'
        subframes = load_video(path)
        print('Video loaded!')
        subframes = torch.FloatTensor(subframes).cuda()
        subframes = subframes[:S]  # picking the first S subframes

    height, width = subframes.shape[1], subframes.shape[2]

    # nbhds = [[(2*i, 2*j), (2*i, 2*j+1), (2*i+1, 2*j), (2*i+1, 2*j+1)] \
    # for i in range(int(height / 2)) for j in range(int(width / 2))]

#    nbhds_rows = torch.LongTensor([[el[0] for el in nbhd] for nbhd in nbhds]).cuda()
#    nbhds_cols = torch.LongTensor([[el[1] for el in nbhd] for nbhd in nbhds]).cuda()

    start = time()
    # c2b_frame_bucket0, c2b_frame_bucket1 = multiplex_v3(subframes, W)
    c2b_frame_bucket0, c2b_frame_bucket1 = multiplex_v5(subframes, (2, 2))
    end = time()

    print(f'Simulation took {end - start} seconds')
    print(c2b_frame_bucket0)
    print(c2b_frame_bucket1)