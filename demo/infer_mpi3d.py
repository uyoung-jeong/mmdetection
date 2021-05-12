# run in base directory.
# e.g.) python demo/infer_mpi3d.py
import torch, torchvision
import mmdet
print(mmdet.__version__)

import mmcv
#from mmcv.ops import get_compiling_cuda_version, get_compiler_version
#print(get_compiling_cuda_version())
#print(get_compiler_version())

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core.visualization import imshow_det_bboxes

import numpy as np
from argparse import ArgumentParser
import os
from os import path

from utils_mpi3d import infer_mpi3d_pipeline

from IPython import embed 

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model', default='MRCNN', choices=['MRCNN', 'Cascade_MRCNN'])
    parser.add_argument('--config', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--device', default=0)

    parser.add_argument('--mpi_base_dir', default='/home/uyoung/human_pose_estimation/datasets/mpi_inf_3dhp')
    args = parser.parse_args()

    if args.model == 'MRCNN':
        #args.config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
        args.config = 'configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py'
        #args.checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
        args.checkpoint = 'checkpoints/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth' 
    elif args.model == 'Cascade_MRCNN':
        args.config = 'configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py'
        args.checkpoint = 'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth'

    args.device = torch.device(args.device)
    return args

def run(args): 
    score_thr = 0.3
    # initialize the detector
    model = init_detector(args.config, args.checkpoint, device=args.device)

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
    
    vis_dir = 'mpi_vis_' + args.model
    
    infer_mpi3d_pipeline(args, vis_dir, infer_fn, model, kwargs=None)

    """
    # Use the detector to do inference
    img = 'demo/demo.jpg'
    result = inference_detector(model, img) # either (bbox, segm) or just bbox

    # Let's plot the result
    show_result_pyplot(model, img, result, score_thr=0.3)
    """

if __name__ == '__main__':
    args = get_args()
    run(args)
