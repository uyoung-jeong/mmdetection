# This code generates segmentation map from preprocessed body part patches
# Currently works only on MPI-INF-3DHP

import os
from os import path
import glob
import json
import scipy.io as sio
import h5py
import shutil

from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
import numpy as np

import matplotlib.pyplot as plt
import cv2

import ipdb
from IPython import embed

# joint names
mpi_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis', 'neck', 'head', 'head_top', 'Lclavicle', 'Lshoulder', 
                   'Lelbow', 'Lwrist', 'Lhand', 'Rclavicle', 'Rshoulder', 'Relbow', 'Rwrist', 'Rhand', 'Lhip', 'Lknee', 
                   'Lankle', 'Lfoot', 'Ltoe', 'Rhip', 'Rknee', 'Rankle', 'Rfoot', 'Rtoe']


# joint indices for direct comparison
# Lshoulder, Lelbow, Lwrist, Rshoulder, Relbow, Rwrist, Lhip, Lknee, Lankle, Rhip, Rknee, Rankle
mpi_comp_keypoints = [9, 10, 11, 14, 15, 16, 18, 19, 20, 23, 24, 25] # index from 0

mpi_17_keypoints_train = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16] # train data
mpi_17_keypoints_test = [14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4] # test data

joints_map_idxs = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6] # map from raw to processed output

vid_list = list(range(3)) + list(range(4,9)) # 0,1,2,4,5,6,7,8

h, w = 2048, 2048 # image height, width

# define body parts
limbs = [[1,2], [2,3], [4,5], [5,6], #legs
        [11,12], [12,13], [14,15], [15,16]] # arms
heads = [[8,9], [9, 10]] # neck, head

#torsos = [[8,1,0,4], [0,14,8,11]] # lower, upper torso
torsos = [[1,0,4,14,8,11]]

body_parts = limbs + heads + torsos

limb_thick = 0.1 # coefficient to make width of the limb parts.
bone_margin_width = 10 # additional length added to the bone length
bone_margin_length = 0.22

rotmat_90 = np.array([[0, -1, 0],
             [1, 0, 0],
             [0, 0, 1]]) # rotation by 90 degree

# load MPI dataset
mpi_base_dir = '/home/uyoung/human_pose_estimation/datasets/mpi_inf_3dhp'
mpi_train_subjects = [f'S{i+1}' for i in range(8)]
mpi_test_subjects = [f'TS{i+1}' for i in range(6)]
seqs = ['Seq1', 'Seq2']

mpi_train_mats = dict()
mpi_train_cams = dict()

for subj in tqdm(mpi_train_subjects, desc='mpi_train_annotations'):
    mpi_train_mats[subj] = dict()
    mpi_train_cams[subj] = dict()
    for seq in seqs:
        mpi_train_cams[subj][seq] = dict()
        anno_path = path.join(mpi_base_dir, subj, seq, 'annot.mat')
        #f = h5py.File(anno_path, 'r') this doesn't work
        data = sio.loadmat(anno_path) # ['__header__', '__version__', '__globals__', 'annot2', 'annot3', 'cameras', 'frames', 'univ_annot3']

        # mpi_train_mats['S1']['Seq1']['annot2'].shape: (14,1)
        # mpi_train_mats['S1']['Seq1']['annot2'][0,0].shape: (6416, 56)
        # mpi_train_mats['S6']['Seq2']['annot2'][0,0].shape: (6145, 56)

        mpi_train_mats[subj][seq] = data

        # get camera calibration data
        cam_path = path.join(mpi_base_dir, subj, seq, 'camera.calibration')
        with open(cam_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()

            i=0
            while i < len(raw_lines):
                tokens = raw_lines[i].strip().split()
                if tokens[0] == 'name':
                    intrinsic_tokens = raw_lines[i+4].strip().split()[1:-4]
                    extrinsic_tokens = raw_lines[i+5].strip().split()[1:]

                    intrinsic = np.array([float(e) for e in intrinsic_tokens]).reshape((3,4))
                    extrinsic = np.array([float(e) for e in extrinsic_tokens]).reshape((4,4))
                    mpi_train_cams[subj][seq]['intrinsic'] = intrinsic
                    mpi_train_cams[subj][seq]['extrinsic'] = extrinsic
                    i += 7
                else:
                    i += 1

mpi_test_mats = dict()
for subj in tqdm(mpi_test_subjects, desc='mpi_test_annotations'):
    anno_path = path.join(mpi_base_dir, 'mpi_inf_3dhp_test_set', subj, 'annot_data.mat')

    f = h5py.File(anno_path, 'r') # ['#refs#', 'activity_annotation', 'annot2', 'annot3', 'bb_crop', 'univ_annot3', 'valid_frame']

    # mpi_test_mats['TS1']['annot2'].shape: (6151, 1, 17, 2)

    mpi_test_mats[subj] = f


parts = []

min_abs_depth = 9999.9 # 1894.76
min_depth = 9999.9 # 1894.76
max_abs_depth = 1.1 # 5684.34
max_depth = 1.1 # 5684.34

# iterate over annotations and draw polygons
vis_dir = 'mpi_vis'
pbar = tqdm(total=len(mpi_train_subjects) * 2, desc='processing mpi train')
for subj in mpi_train_subjects: # per subject
    subj_dir = path.join(mpi_base_dir, subj)
    if not path.exists(path.join(vis_dir, subj)):
        os.makedirs(path.join(vis_dir, subj))
    for seq in seqs: # per seq
        seq_dir = path.join(subj_dir, seq)
        for j, vid_i in enumerate(vid_list): # per video
            # image folder. requires preprocessing procedure in SPIN
            video_dir = path.join(seq_dir, f'video_{vid_i}')

            img_list = glob.glob(path.join(video_dir, '*.jpg'))
            img_list.sort()
            for i, img_i in enumerate(img_list): # per frame
                if i % 2000 != 0: # process per every 100 frames
                    continue
                """
                if i > 0:
                    break
                """
                
                img = cv2.imread(img_i)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_name = img_i.split('/')[-1]

                joints = np.reshape(mpi_train_mats[subj][seq]['annot2'][vid_i][0][i], (28,2))[mpi_17_keypoints_train]
                joints_3d = np.reshape(mpi_train_mats[subj][seq]['annot3'][vid_i][0][i], (28,3))[mpi_17_keypoints_train]
                
                # check that all joints are visible
                x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
                y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
                ok_pts = np.logical_and(x_in, y_in)
                if np.sum(ok_pts) < len(joints_map_idxs): # skip partly/totally invisible cases
                    continue

                # get depth values
                # x = w(x,y,1)^T = PX
                intr = mpi_train_cams[subj][seq]['intrinsic']
                extr = mpi_train_cams[subj][seq]['extrinsic']

                joints_proj = np.matmul(intr, np.vstack((joints_3d.T, np.ones(17))))
                depths = joints_proj[-1,:]
                dist_factor = np.sqrt(np.mean(depths) / 1000) # depth distance factor

                """
                mapped_joints = np.zeros([24, 3]) # why 24?
                mapped_joints[joints_map_idxs] = np.hstack([joints, np.ones([17, 1])])
                parts.append(mapped_joints)
                """
                
                """
                # visualize keypoints
                vis_dir = 'mpi_vis'
                if not path.exists(vis_dir):
                    os.makedirs(vis_dir)
                for j_idx, joint in enumerate(joints):
                    plt.imshow(img_rgb)
                    plt.scatter(joint[0], joint[1], s=15)
                    plt.savefig(path.join(vis_dir, f'{subj}_{seq}_{vid_i}_{i}_{j_idx}.png'))
                    plt.clf()
                plt.imshow(img_rgb)
                plt.scatter(joints[:,0], joints[:,1])
                plt.savefig(path.join(vis_dir, f'{subj}_{seq}_{vid_i}_{i}_total.png'))
                plt.clf()
                """

                # make patches
                seg_map = np.zeros(img.shape[:-1]).astype(np.uint8) # single channel
                for part_i, part in enumerate(body_parts):
                    s,t = joints[part[0]], joints[part[1]]
                    s1, s2, t1, t2 = None, None, None, None
                    """
                    if len(part)==2:
                        # get perpendicular vector by 90 degree rotation and then apply scaling
                        dist_factor = np.sqrt(np.mean(depths) / 1000) # depth distance factor
                        perp_v = (2/(dist_factor) * limb_thick * np.matmul(rotmat_90, np.hstack((t-s, np.array([1]))))
                        
                        # validate the perpendicular vector
                        plt.plot([s[0], t[0]], [s[1], t[1]])
                        s2 = s + perp_v[:-1]
                        plt.plot([s[0], s2[0]], [s[1], s2[1]])
                        plt.savefig(path.join(vis_dir, 'perp.png'))
                        perp_v = perp_v[:-1] # make it inhomogeneous

                        # get 4 polygon points
                        s1 = s - perp_v
                        s2 = s + perp_v
                        t1 = t - perp_v
                        t2 = t + perp_v
                    else: # torso case
                        s1, s2, t2, t1 = joints[part[0]], joints[part[1]], joints[part[2]], joints[part[3]]
                        
                    polygon = np.vstack((s1, s2, t2, t1))
                    """
                    if len(part)==2:
                        # fill ellipse
                        ellipse_center = (s+t)/2 # need to be int
                        ellipse_center = tuple([int(e) for e in ellipse_center])
                        bone_norm = np.linalg.norm(t-s)
                        axes_length = [bone_norm, 2/dist_factor * limb_thick * bone_norm + bone_margin_width] # x, y length. x: bone length. y: width
                        
                        #if part_i < 4: # if leg, reduce bone length
                        axes_length[0] = axes_length[0] * (1-bone_margin_length)
                        
                        axes_length = tuple([int(e) for e in axes_length])
                        bone_unit_vector = (t-s)/bone_norm
                        angle = np.arctan2(bone_unit_vector[1], bone_unit_vector[0])# rotation angle to align with t-s vector
                        angle = np.rad2deg(angle) # convert from radian to degree
                        start_angle = 0
                        end_angle = 360
                        ellipse_color = (255,255,255)
                        thickness = -1
    
                        seg_map = cv2.ellipse(seg_map, ellipse_center, axes_length, angle, start_angle, end_angle, ellipse_color, thickness)
                    else: # torso case. fill the polygon
                        #s1, s2, t2, t1 = joints[part[0]], joints[part[1]], joints[part[2]], joints[part[3]]
                        #polygon = np.vstack((s1, s2, t2, t1)) # [4,2]
                        
                        pts = joints[part]
                        
                        seg_map = cv2.fillPoly(seg_map, pts=np.array([pts], dtype=np.int32), color=(255,255,255))
                    """
                    # fill the polygon
                    try:
                        seg_map = cv2.fillPoly(seg_map, pts=np.array([polygon], dtype=np.int32), color=(255,255,255))
                    except cv2.error as e:
                        print(e)
                        ipdb.set_trace()
                        print()
                    """

                cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg.png'), seg_map) # save seg map
                seg_map = np.clip(seg_map, 0, 1)

                seg_repeat = np.repeat(seg_map.reshape((*seg_map.shape, 1)), 3, axis=-1)
                cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg_orig.png'), np.multiply(img, seg_repeat)) # seg_map * img
                cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_orig.png'), img) # original img

                parts.append(joints)
        pbar.update(1)
    #print()
#print()
pbar.close()


