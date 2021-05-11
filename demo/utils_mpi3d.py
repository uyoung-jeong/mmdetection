import os
from os import path
import glob

import cv2
import numpy as np
import h5py
import scipy.io as sio
from tqdm import tqdm
    
# work as a wrapper of inference pipeline
# infer_fn: gets raw image as an input, and returns [0,1] range seg. map
# kwargs: additional arguments for infer_fn
def infer_mpi3d_pipeline(args, vis_dir, infer_fn, model, kwargs=None):
    # joint names
    mpi_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis', 'neck', 'head', 'head_top', 'Lclavicle', 'Lshoulder', 
                       'Lelbow', 'Lwrist', 'Lhand', 'Rclavicle', 'Rshoulder', 'Relbow', 'Rwrist', 'Rhand', 'Lhip', 'Lknee', 
                       'Lankle', 'Lfoot', 'Ltoe', 'Rhip', 'Rknee', 'Rankle', 'Rfoot', 'Rtoe']

    # joint indices for direct comparison
    # Lshoulder, Lelbow, Lwrist, Rshoulder, Relbow, Rwrist, Lhip, Lknee, Lankle, Rhip, Rknee, Rankle
    mpi_comp_keypoints = [9, 10, 11, 14, 15, 16, 18, 19, 20, 23, 24, 25] # index from 0

    # Pelvis, Lhip, Lknee, Lankle, Rhip, Rknee, Rankle, Spine1, Spine2(somewhat middle of shoulder lines), neck top, 
    # head top, Lshoulder, Lelbow, Lwrist, Rshoulder, Relbow, Rwrist
    mpi_17_keypoints_train = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16] # train data
    mpi_17_keypoints_test = [14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4] # test data

    joints_map_idxs = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6] # map from raw to processed output

    vid_list = list(range(3)) + list(range(4,9)) # 0,1,2,4,5,6,7,8

    h, w = 2048, 2048 # image height, width

    # load MPI dataset
    #mpi_base_dir = '/home/uyoung/human_pose_estimation/datasets/mpi_inf_3dhp'
    mpi_base_dir = args.mpi_base_dir
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


    # iterate over each image
    #vis_dir = 'mpi_vis'
    
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
                    if i % 1000 != 0: # process per every 100 frames
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
                        
                    # inference process
                    seg_map, det_res = infer_fn(model, img, kwargs)
                    
                    cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg.png'), seg_map) # save seg map
                    cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg_255.png'), seg_map * 255) # save seg map
    
                    seg_repeat = np.repeat(seg_map.reshape((*seg_map.shape, 1)), 3, axis=-1)
                    cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg_orig.png'), np.multiply(img, seg_repeat)) # seg_map * img
                    cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_orig.png'), img) # original img
    
                    #parts.append(joints)
            pbar.update(1)
    pbar.close()