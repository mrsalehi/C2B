from c2b import C2BCamera
import argparse
import yaml
from functools import partial
from utils import img_seq_gen, video_gen


# Things to be added:


# 1. Voltage threshold



# 2. Support for loading a video file and 


# 3. Support for predicting the slope of the charging line

def main(args, config):
    if config.data.type == 'img_seq':
        gen_fun = partial(img_seq_gen, config.data.path, config.data.extension)
    elif config.data.type == 'video':    
        gen_fun = partial(video_gen, config.data.path)
        
    c2b = C2BCamera(args, config, gen_fun)
    c2b.get_c2b_frames_from_preexisting_frames()


def get_args_config():
    parser = argparse.ArgumentParser(description='C2B Camera Simulation')    
    parser.add_argument('--config', type=str, default="configs/default.yaml", help='Path to config file')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to run the simulation on')
    
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
            
        return namespace

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = dict2namespace(config)
    
    return args, config


if __name__ == "__main__":
    args, config = get_args_config()
    main(args, config)