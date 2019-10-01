# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import glob
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc
import imageio

import config
import training.misc as misc
import dnnlib.tflib.tfutil as tfutil

from PIL import Image

def random_latents(num_latents, G, random_state=None):
    if random_state is not None:
        return random_state.randn(num_latents, *G.input_shape[1:]).astype(np.float32)
    else:
        return np.random.randn(num_latents, *G.input_shape[1:]).astype(np.float32)

#----------------------------------------------------------------------------
# Generate random images or image grids using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_fake_images(run_id, snapshot=None, grid_size=[1,1], num_pngs=1, image_shrink=1, subdir=None, random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if subdir is None:
        subdir = misc.get_id_string_for_network_pkl(network_pkl)
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    result_subdir = "results/images/" + subdir
    if not os.path.exists(result_subdir):
        os.makedirs(result_subdir)
    for png_idx in range(num_pngs):
        print('Generating png %d / %d...' % (png_idx, num_pngs))
        latents = random_latents(np.prod(grid_size), Gs, random_state=random_state)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        misc.save_image_grid(images, os.path.join(result_subdir, '%06d.png' % (png_idx)), [0,255], grid_size)
        np.save(result_subdir + "/" + '%06d' % (png_idx), latents)

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_interpolation_video(run_id, snapshot=None, grid_size=[1,1], image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)

    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)
    if mp4 is None:
        mp4 = misc.get_id_string_for_network_pkl(network_pkl) + '-seed-' + str(random_seed) + '.avi'

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    print(shape)
    print(len(shape))
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))
    print(all_latents[0].shape)


    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = "results/videos"
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='png', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_keyframed_video(run_id, latents_idx, subdir=None, snapshot=None, grid_size=[1,1], image_shrink=1, image_zoom=1, transition_frames = 25, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    if subdir is None:
        subdir = misc.get_id_string_for_network_pkl(network_pkl)
    keyframe_dir = "results/images/" + subdir
    result_subdir = "results/videos/" + subdir
    if not os.path.exists(result_subdir):
        os.makedirs(result_subdir)

   # codec mp4
         
   # if mp4 is None:
   #     count = len(glob.glob(result_subdir + "/*.avi"))
   #     mp4 = str(count) + '-video.avi'

   # codec AVI

    if mp4 is None:
        count = len(glob.glob(result_subdir + "/*.avi"))
        mp4 = str(count) + '-video.avi'

    files = [f for f in glob.glob(keyframe_dir + "/*.npy", recursive=True)]

    for f in files:
        print(f)

    latents = list(map(lambda idx: np.load(files[idx]), latents_idx))
    print('len(latents)', len(latents))
    
    num_frames = transition_frames * len(latents)
    duration_sec = num_frames / mp4_fps

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))

        section = frame_idx // transition_frames

        start = latents[section]
        end = latents[(section + 1) % len(latents)]

        transition_i = frame_idx - section * transition_frames
        maxindex = transition_frames-1.0
        mu1 = min(max(0, (transition_i*1.0/maxindex)*(transition_i*1.0/maxindex) ), 1)
       #mu1 = min(max(0, (transition_i*1.0/maxindex) ), 1)
        lat = np.multiply(start, 1.0-mu1)+ np.multiply(end, mu1)
        labels = np.zeros([lat.shape[0], 0], np.float32)
        images = Gs.run(lat, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='png', bitrate=mp4_bitrate)
    with open(os.path.join(result_subdir, mp4 + '-keyframes.txt'), 'w') as file:
        file.write(str(latents_idx))

## -----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    import time
    print(datetime.datetime.now(), int(time.time()))
    np.random.seed(int(time.time()))
    tfutil.init_tf()
   # generate_fake_images(12, num_pngs=100)
   # generate_interpolation_video(12, grid_size=[1,1], random_seed=int(time.time()), mp4_fps=25, duration_sec=300.0)
    keyframes = [11,36,71,74,94]
    generate_keyframed_video(12, keyframes)
    print('Exiting...')
    print(datetime.datetime.now())
