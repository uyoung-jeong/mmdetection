# This code generates segmentation map from preprocessed body part patches
# Currently works only on MPI-INF-3DHP
# python demo/generate_seg_ensemble.py --vis_dir /home/uyoung/human_pose_estimation/datasets/mpi_inf_3dhp --frame_freq 10

import os
from os import path
import glob
import scipy.io as sio
import h5py
from tqdm import tqdm

import torch
import numpy as np
import cv2
from argparse import ArgumentParser

import mmdet
import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core.visualization import imshow_det_bboxes

from utils_mpi3d import infer_mpi3d_pipeline

import ipdb
from IPython import embed

class Config():
    def __init__(self):
        self.mrcnn_config = 'configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py'
        self.mrcnn_checkpoint = 'checkpoints/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth'
        self.cascade_mrcnn_config = 'configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py'
        self.cascade_mrcnn_checkpoint = 'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth'
        self.device = torch.device(0)
        self.mpi_base_dir = ''
        self.vis_dir = 'mpi_vis_ensemble'
        self.frame_freq = 1000

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--device', default=0)
    parser.add_argument('--mpi_base_dir', default='/home/uyoung/human_pose_estimation/datasets/mpi_inf_3dhp')
    parser.add_argument('--vis_dir', default='mpi_vis_ensemble')
    parser.add_argument('--frame_freq', default=1000, type=int)
    args = parser.parse_args()

    config = Config()
    config.device = torch.device(args.device)
    config.mpi_base_dir = args.mpi_base_dir
    config.vis_dir = args.vis_dir
    config.frame_freq = args.frame_freq
    return config

def infer_mpi3d_pipeline(args, vis_dir, infer_fn, models, kwargs=None):
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
    margin = 200

    # define body parts
    limbs = [[1,2], [2,3], [4,5], [5,6], #legs
            [11,12], [12,13], [14,15], [15,16]] # arms
    heads = [[8,9], [9, 10]] # neck, head

    #torsos = [[8,1,0,4], [0,14,8,11]] # lower, upper torso
    torsos = [[1,0,4,14,8,11]]

    body_parts = limbs + heads + torsos

    limb_thick = 0.1 # coefficient to make width of the limb parts.
    bone_margin_width = 10 # additional length added to the bone length
    bone_margin_length = 0.3

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
    pbar = tqdm(total=len(mpi_train_subjects) * 2, desc='processing mpi train')
    for subj in mpi_train_subjects: # per subject
        subj_dir = path.join(mpi_base_dir, subj)
        if not path.exists(path.join(vis_dir, subj)):
            os.makedirs(path.join(vis_dir, subj))
        for seq in seqs: # per seq
            seq_dir = path.join(subj_dir, seq)
            # get depth values
            # x = w(x,y,1)^T = PX
            intr = mpi_train_cams[subj][seq]['intrinsic']
            extr = mpi_train_cams[subj][seq]['extrinsic']

            for j, vid_i in enumerate(vid_list): # per video
                # image folder. requires preprocessing procedure in SPIN
                video_dir = path.join(seq_dir, f'video_{vid_i}')

                img_list = glob.glob(path.join(video_dir, '*.jpg'))
                img_list.sort()
                for i, img_i in enumerate(img_list): # per frame
                    if i < 100 and subj == 'S2' and seq=='Seq1': # S2, Seq1, frame 0 has incorrect gt
                        continue
                    if i % args.frame_freq != 0: # process per every frame_freq frames
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

                    # filter out non-human area. note that x,y positions are swapped
                    xmin = np.min(joints[:,1])
                    xmax = np.max(joints[:,1])
                    ymin = np.min(joints[:,0])
                    ymax = np.max(joints[:,0])
                    bbox_center = [(xmin+xmax)/2,(ymin+ymax)/2]
                    bbox_len = (np.max([xmax-xmin,ymax-ymin]) + 400)/2
                    bbox = [np.max([bbox_center[0]-bbox_len, 0]),
                            np.min([bbox_center[0]+bbox_len, w]),
                            np.max([bbox_center[1]-bbox_len, 0]),
                            np.min([bbox_center[1]+bbox_len, h])]
                    bbox = np.array(bbox).astype(int)

                    img_filtered = np.zeros(img.shape).astype(np.uint8)
                    img_filtered[bbox[0]:bbox[1], bbox[2]:bbox[3]] = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]

                    # inference process
                    seg_maps = []
                    seg_map = None
                    for model in models:
                        seg_map, det_res = infer_fn(model, img_filtered, kwargs)
                        if seg_map is not None:
                            seg_maps.append(seg_map)

                    if len(seg_maps) == 0: # if no map is inferred,
                        pass
                    elif len(seg_maps) == 1: # single seg_map
                        seg_map = seg_maps[0]
                    elif len(seg_maps) > 1: # 2 seg_maps
                        seg_map = np.logical_and(seg_maps[0], seg_maps[1]).astype(np.uint8)

                    #if seg_map is not None:
                    #    cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg_model.png'), seg_map*255) # save seg map

                    ### get 2D seg_patch
                    joints_proj = np.matmul(intr, np.vstack((joints_3d.T, np.ones(17))))
                    depths = joints_proj[-1,:]
                    dist_factor = np.sqrt(np.mean(depths) / 1000) # depth distance factor

                    # make patches
                    seg_patch = np.zeros(img.shape[:-1]).astype(np.uint8) # single channel
                    for part_i, part in enumerate(body_parts):
                        s,t = joints[part[0]], joints[part[1]]
                        s1, s2, t1, t2 = None, None, None, None

                        if len(part)==2:
                            # fill ellipse
                            ellipse_center = (s+t)/2 # need to be int
                            ellipse_center = tuple([int(e) for e in ellipse_center])
                            bone_norm = np.linalg.norm(t-s)
                            axes_length = [bone_norm, 2/dist_factor * limb_thick * bone_norm + bone_margin_width] # x, y length. x: bone length. y: width

                            #if part_i < 4: # if leg, reduce bone length
                            if part_i != 9: # if not head:
                                axes_length[0] = axes_length[0] * (1-bone_margin_length)
                            else:
                                axes_length[0] = axes_length[0] * (1-0.45)

                            axes_length = tuple([int(e) for e in axes_length])
                            bone_unit_vector = (t-s)/bone_norm
                            angle = np.arctan2(bone_unit_vector[1], bone_unit_vector[0])# rotation angle to align with t-s vector
                            angle = np.rad2deg(angle) # convert from radian to degree
                            start_angle = 0
                            end_angle = 360
                            ellipse_color = (255,255,255)
                            thickness = -1

                            seg_patch = cv2.ellipse(seg_patch, ellipse_center, axes_length, angle, start_angle, end_angle, ellipse_color, thickness)
                        else: # torso case. fill the polygon
                            #s1, s2, t2, t1 = joints[part[0]], joints[part[1]], joints[part[2]], joints[part[3]]
                            #polygon = np.vstack((s1, s2, t2, t1)) # [4,2]

                            pts = joints[part]

                            seg_patch = cv2.fillPoly(seg_patch, pts=np.array([pts], dtype=np.int32), color=(255,255,255))

                    #cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg_patch.png'), seg_patch) # save seg map

                    if seg_map is None: # if no seg_map is inferred, just use patch
                        seg_map = seg_patch
                    else: # otherwise, perform logical or operation btw. inferred map and the patch
                        seg_map = np.logical_or(seg_map, seg_patch)

                    if seg_map is None:
                        print(f'{subj}/{seq}/{vid_i}_{i}: could not make seg_map')
                        continue

                    seg_map = np.clip(seg_map.astype(np.uint8),0,1) # reformat and clip

                    # make directory
                    seg_vid_dir = path.join(vis_dir, subj, seq, f'video_{vid_i}_seg')
                    if not path.exists(seg_vid_dir):
                                os.makedirs(seg_vid_dir)
                    frame_idx = str(i).zfill(6)
                    cv2.imwrite(path.join(seg_vid_dir, f'{frame_idx+1}.png'), seg_map)
                    #cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg.png'), seg_map) # save final seg map
                    #cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg_255.png'), seg_map * 255) # save seg map

                    #seg_repeat = np.repeat(seg_map.reshape((*seg_map.shape, 1)), 3, axis=-1)
                    #cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_seg_orig.png'), np.multiply(img, seg_repeat)) # seg_map * img
                    #cv2.imwrite(path.join(vis_dir, f'{subj}', f'{seq}_{vid_i}_{i}_orig.png'), img) # original img

                    #parts.append(joints)
            pbar.update(1)
    pbar.close()

def run(args):
    score_thr = 0.3
    # initialize the detector
    models = [init_detector(args.mrcnn_config, args.mrcnn_checkpoint, device=args.device),
              init_detector(args.cascade_mrcnn_config, args.cascade_mrcnn_checkpoint, device=args.device)]

    def infer_fn(model, img, kwargs=None):
        result = inference_detector(model, img)

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # get person index
        person_idx = 0
        if isinstance(model.CLASSES, tuple):
            person_idx = model.CLASSES.index('person')


        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        # filter out results by threshold
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes_ = bboxes[inds, :]
        labels_ = labels[inds]
        segms_ = None
        if segms is not None:
            segms_ = segms[inds, ...]
        else:
            return None, None

        # aggregate person indices
        seg_map = segms_[labels_==person_idx] * 1

        if seg_map.size == 0: # no segmap is generated
            return None, None

        seg_map = np.max(seg_map, axis=0)
        seg_map = np.clip(seg_map * 1, 0, 1)
        seg_map = seg_map.astype(np.uint8)

        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=model.CLASSES,
            score_thr=score_thr,
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),
            mask_color=None,
            thickness=2,
            font_size=13,
            win_name='',
            show=False,
            wait_time=0,
            out_file=None)
        return seg_map, img

    infer_mpi3d_pipeline(args, args.vis_dir, infer_fn, models, kwargs=None)

if __name__ == '__main__':
    args = get_args()
    run(args)
