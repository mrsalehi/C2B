import torch
import cv2
import PIL
from typing import List
from utils import load_video
from time import time


S = 5


def multiplex_v3(subframes: torch.Tensor, W: torch.Tensor):
    """
    subframes: intensities of subframes of a frame as a numpy array with shape (H, W, S)
    W: numpy array with shape (nbhd_height, nbhd_width, S)
    nbhd_size: tuple of width and height of the neighborhood (For now W must be divisible by the width of the nbhd 
    and H must be divisible by the height of the neighborhood)
    """
    height, width = subframes.shape[0], subframes.shape[1]
    # c2b_frame_bucket0 = torch.zeros(height, width).cuda()
    # c2b_frame_bucket1 = torch.zeros(height, width).cuda()
    
    nbhd_height, nbhd_width = W.shape[0], W.shape[1]

    assert height % nbhd_height == 0
    assert width % nbhd_width == 0

    W_ = torch.tile(W, (height // nbhd_height, width // nbhd_width))

    c2b_frame_bucket0 = torch.sum(subframes * W_, dim=2)  # shape: (H, W)
    c2b_frame_bucket1 = torch.sum(subframes * (1 - W_), dim=2)  # shape: (H, W)

    return c2b_frame_bucket0, c2b_frame_bucket1


def multiplex_v2(subframes: torch.Tensor, W: torch.Tensor, nbhds_rows: torch.Tensor, \
    nbhds_cols: torch.Tensor):
    """
    subframes: intensities of subframes of a frame as a numpy array with shape (S, W, H)
    W: numpy array with shape (neighborhood_size, S)
    """
    height, width = subframes.shape[1], subframes.shape[2]
    c2b_frame_bucket0 = torch.zeros(height, width).cuda()
    c2b_frame_bucket1 = torch.zeros(height, width).cuda()

    nbhd_size = len(nbhds_rows[0])  # size of the neighborhoods (number of pixels in them)
    n_nbhds = len(nbhds_rows)  # number of neightborhoods 
    
    start = time()
    nbhd_subframes = [subframes[:, rows, cols] for rows, cols in zip(nbhds_rows, nbhds_cols)]  # list of tensors with shape (S, nbhd_size)
    end = time()
    print(f'First for loop took {end - start} seconds')

    nbhd_subframes = torch.stack(nbhd_subframes, dim=0)  # shape: (n_nbhds, S, neighborhood_size)

    # sample one intensity from each and every subframe of the neighborhoods
    start = time()
    nbhd_subframes_samples = torch.gather(nbhd_subframes, 2, torch.randint(nbhd_size, size=(n_nbhds, S, 1)).cuda())  # shape: (n_nbhds, S, 1)
    end = time()
    print(f'Gather took {end - start} seconds')
    
    c2b_nbhds_bucket0 = torch.matmul(nbhd_subframes_samples.permute(0, 2, 1), W.T).squeeze()  # shape: (nbhd_size,)
    c2b_nbhds_bucket1 = torch.matmul(nbhd_subframes_samples.permute(0, 2, 1), 1 - W.T).squeeze()  # shape: (nbhd_size,)
    print(f'Matrix multiplication took {end - start} seconds')

    start = time()
    for i, (row, col) in enumerate(zip(nbhds_rows, nbhds_cols)):
        c2b_frame_bucket0[row, col] = c2b_nbhds_bucket0[i]
        c2b_frame_bucket1[row, col] = c2b_nbhds_bucket1[i]
    end = time()
    print(f'Second for loop took {end - start} seconds')

    #c2b_frame_bucket0[nbhd_row, nbhd_col] = c2b_nbhd_bucket0
    #c2b_frame_bucket1[nbhd_row, nbhd_col] = c2b_nbhd_bucket1

    return c2b_frame_bucket0, c2b_frame_bucket1


def multiplex(subframes: torch.Tensor, W: torch.Tensor, nbhds: List[List[tuple]]):
    """
    subframes: intensities of subframes of a frame as a numpy array with shape (S, W, H)
    W: numpy array with shape (neighborhood_size, S)
    nbhds: neighborhoods 
    """
    height, width = subframes.shape[1], subframes.shape[2]
    c2b_frame_bucket0 = torch.zeros(height, width)
    c2b_frame_bucket1 = torch.zeros(height, width)

    for nbhd in nbhds:
        nbhd_row = [el[0] for el in nbhd]
        nbhd_col = [el[1] for el in nbhd]
        nbhd_size = len(nbhd_row)
        nbhd_subframes = subframes[:, nbhd_row, nbhd_col]  # shape: (S, nbhd_size)
        nbhd_subframes_samples = nbhd_subframes[torch.arange(S), torch.randint(nbhd_size, size=(S,))]  # shape: (S,)
        c2b_nbhd_bucket0 = torch.matmul(nbhd_subframes_samples, W.T) # shape: (nbhd_size,)
        c2b_nbhd_bucket1 = torch.matmul(nbhd_subframes_samples, 1 - W.T) # shape: (nbhd_size,)
        
        c2b_frame_bucket0[nbhd_row, nbhd_col] = c2b_nbhd_bucket0
        c2b_frame_bucket1[nbhd_row, nbhd_col] = c2b_nbhd_bucket1

    return c2b_frame_bucket0, c2b_frame_bucket1


def demultiplex():
    pass

# def get_bucket_measurements(input: np.ndarray):
#     if input.ndim  == 3:
#         # image
#         mask = np.random.randint(low=0, high=2, size=(1, S))  # shape: 1 * S
#         multiplex_mat = np.vstack([mask, 1 - mask])
#         pixel_intensities = input[1]  # using green channel as the intensity
#         bucket_measure = multiplex_mat @ pixel_intensities
#     elif input.ndim == 4:
#         # video
#         mask = np.random.randint(low=0, high=2, size=(input.shape[0], S))  # shape: F * S
#         multiplex_mat = np.vstack([mask, 1 - mask])
#         bucket_measure = []
#         for frame in input:
#             pixel_intensities = input[1]
#             bucket_measure.append(multiplex_mat @ pixel_intensities)
        
#         bucket_measure = np.stack(bucket_measure)


if __name__ == '__main__':
    DEBUG = 1
    
    W = torch.FloatTensor([[[1, 1, 0], [1, 1, 1]], [[0 ,0, 0], [1, 0, 1]]]).cuda()
    
    if DEBUG:
        subframes = torch.FloatTensor([[[1, 1, 1], [1, 0, 0]], [[0 ,0, 0], [0, 0, 1]]]).cuda()
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
    c2b_frame_bucket0, c2b_frame_bucket1 = multiplex_v3(subframes, W)
    end = time()

    print(f'Simulation took {end - start} seconds')
    print(c2b_frame_bucket0)
    print(c2b_frame_bucket1)