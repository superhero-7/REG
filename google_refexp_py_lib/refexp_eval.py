"""Python class for the evaluation of Google Refexp dataset.

This script contains two python classes:
1. GoogleRefexpEvalComprehension
  -  Use precision@k score to evaluate comprehension task performance
  -  Can evaluate generation task through an end-to-end way
2. GoogleRefexpEvalGeneration
  -  Use Amazon Mechanical Turker (AMT) to compare generated refexps with GT
     with the following steps (step a, c, f covered by the class):
     a. Generate csv files for AMT
     b. Generate images and masked images
     c. Upload these images to a server (e.g. Amazon S3) so that the image are 
        publicly accessible
     d. Create a AMT project with the interface at 
        ./cache_evaluation/AMT_interface/AMT_template_generated_vs_GT.html
     e. Upload csv files and start AMT job
     f. Download annotated json file and calculate the score

TO CHECK:
GoogleRefexp.getAnnIds(): get COCO object ids
GoogleRefexp.getRefexpIds(): get referring expression ids
GoogleRefexp.getRefexpAnns(): get google refexp annotations for a list of annotation_id
GoogleRefexp.getGtBoxes(): currently assume a dictionary with key of id, value of a list for bbox

TODO:
Comprehention:
-  A script that can visualize predicted bboxes whose iou satistied a constrain
"""

import json
import os
import copy
import random
import sys
import numpy
import csv
from scipy import misc
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from refexp import Refexp # Need to check - change
import common_utils as cu

class RefexpEvalComprehension(object):
  def __init__(self, refexp_dataset_path, coco_data_path):
    """Constructor for GoogleRefexpEvalComprehension class for evaluation.
    
    Args:
      refexp_dataset_path: path for the Google Refexp dataset file
      coco_data_path: path for the original coco dataset file (e.g. 'instances_train2014.json')
    """
    # handle refexp dataset file
    assert refexp_dataset_path, "Refexp dataset file missing!"
    self.refexp_dataset_path = refexp_dataset_path
    print ('Loading Google Refexp dataset file for the comprehension task.')
    self.refexp_dataset = Refexp(refexp_dataset_path, coco_data_path) # Need to check - change
    self.gt_ann_ids_set = frozenset(self.refexp_dataset.getAnnIds()) # Need to check - change
    self.gt_refexp_ids_set = frozenset(self.refexp_dataset.getRefexpIds()) # Need to check - change
    
    # reset evaluation state 重置评估状态
    self.reset_eval_state()
    
  def reset_eval_state(self):
    """Reset evaluation state."""
    self.pred_results_path = None
    self.pred_results = None
    self.flag_already_eval = False
    
  def evaluate(self, pred_results_path,
               thresh_iou=0.5,
               thresh_k=1,
               flag_ignore_non_existed_object=False,
               flag_ignore_non_existed_gt_refexp=False,
               flag_missing_objects_verbose=False,
               flag_missing_refexps_verbose=False):
    """Evaluate the predicted results for the comprehension task.
    
    Args:
      pred_results_path: path for the predicted results with the format
          described in ./cache_evaluation/format_comprehension_eval.md
      thresh_iou: threshold of the IoU ratio of the evaluation
      thresh_k: precision@k
      flag_ignore_non_existed_object: if set True, the evaluation process
          continues with an warning when encountered non existed objects in 
          self.refexp_dataset. Otherwise stops.
      flag_ignore_non_existed_gt_refexp: if set True, the evaluation process  
          continues when encountered non existed GT referring expressions.
          Otherwise stops.
      flag_missing_objects_verbose: if set true, will list the ids of all the 
          missing objects in self.refexp_dataset
      flag_missing_refexps_verbose: if set true, will list the ids of all the 
          missing referring expressions in self.refexp_dataset
          
    Returns:
      A two element tuple. The first element is precision@k. The second
      element is the predicted results (a dictionary) with an added field
      'best_iou' of the best iou for the top k bounding boxes.
    """
    # Load predicted results
    self.reset_eval_state()
    print ('Loading predicted result file for the comprehension task.')
    with open(pred_results_path) as fin:
      self.pred_results = json.load(fin)
    
    # evaluation
    pred_ann_ids_set = set()
    pred_refexp_ids_set = set()
    score = 0.0
    num_valid_pred = 0
    for pred_elem in self.pred_results:
      # validate the predicted results
      assert 'annotation_id' in pred_elem, 'Object annotation id missing!'
      assert 'predicted_bounding_boxes' in pred_elem, \
             'list of predicted bounding boxes missing!'
      ann_id = pred_elem['annotation_id']
      gt_bbox = self._get_GT_bbox_with_annotation_id(ann_id) # Need to check - change coco里面的bbox，就是ground-truth啊！！
      if gt_bbox is None:
        if flag_ignore_non_existed_object:
          print ('Ignore COCO annotation id %d which does not exist in '
                 'Refexp dataset file for evaluation' % ann_id)
          pred_elem['best_iou'] = 0.0
          continue
        else:
          print ('COCO annotation id %d does not exist in Refexp '
                 'dataset file for evaluation!' % ann_id)
          raise
      if ('refexp_id' in pred_elem) and not(pred_elem['refexp_id'] in self.gt_refexp_ids_set):
        if flag_ignore_non_existed_gt_refexp:
          print ('Ignore refexp id %d which does not exist in '
                 'Refexp dataset file for evaluation' % pred_elem['refexp_id'])
          pred_elem['best_iou'] = 0.0
          continue
        else:
          print ('refexp id %d does not exist in Refexp '
                 'dataset file for evaluation!' % pred_elem['refexp_id'])
          raise
      pred_ann_ids_set.add(ann_id)
      if 'refexp_id' in pred_elem:
        pred_refexp_ids_set.add(pred_elem['refexp_id'])
      num_valid_pred += 1
          
      # check whether it is a correct prediction
      pred_bboxes = pred_elem['predicted_bounding_boxes']
      best_iou = 0.0
      for k in range(min(thresh_k, len(pred_bboxes))):
        iou = cu.iou_bboxes(pred_bboxes[k], gt_bbox) # cu又是从common utils里引入进来的
        best_iou = max(best_iou, iou)
      if best_iou >= thresh_iou:
        score += 1.0
      pred_elem['best_iou'] = best_iou
    score /= num_valid_pred
      
    # warning for missing objects and refexps
    gt_ann_ids_left_set = self.gt_ann_ids_set - pred_ann_ids_set
    #gt_refexp_ids_left_set = self.gt_refexp_ids_set - pred_refexp_ids_set
    if gt_ann_ids_left_set:
      print ('Missing %d objects in the refexp dataset file in the predicted '
             'file' % len(gt_ann_ids_left_set))
      print (' and the refexp have %d objects totally' % len(self.gt_ann_ids_set))
      if flag_missing_objects_verbose:
        print ('The missing object annotation ids are:')
        print (gt_ann_ids_left_set)  # TODO pretty print format
#     if gt_refexp_ids_left_set:
#       print ('Missing %d refexps in the refexp dataset file in the predicted '
#              'file' % len(gt_refexp_ids_left_set))
#       if flag_missing_refexps_verbose:
#         print ('The missing refexp ids are:')
#         print (gt_refexp_ids_left_set)  # TODO pretty print format
      
    # summarize the results
    print ('The average prec@%d score is %.3f' % (thresh_k, score))
    return (score, self.pred_results)
    
  def _get_GT_bbox_with_annotation_id(self, ann_id):
    if not ann_id in self.gt_ann_ids_set:
      return None
    anns = self.refexp_dataset.loadAnns(ids = [ann_id])
    if len(anns) == 0:
      return None
    assert len(anns) == 1
    return anns[0]['bbox']
    
  def visualize_top_predicted_bbox(self, pred_sample, coco_image_dir):
    """Visualize the top predicted bounding box."""
    assert 'annotation_id' in pred_sample, 'Object annotation id missing!'
    assert 'predicted_bounding_boxes' in pred_sample, \
           'list of predicted bounding boxes missing!'
    if not pred_sample['predicted_bounding_boxes']:
      print ('Empty predicted bounding boxes.')
      return
      
    bbox_pred_top = pred_sample['predicted_bounding_boxes'][0]
    ann_id = pred_sample['annotation_id']
    ann = self.refexp_dataset.loadAnns(ids=[ann_id])[0]
    image_id = ann['image_id']
    img_coco = self.refexp_dataset.loadImgs(ids=[image_id])[0]
    iou = cu.iou_bboxes(bbox_pred_top, ann['bbox'])
    
    if 'refexp' in pred_sample or 'refexp_id' in pred_sample:
      print ('The Referring expression input to the model is:')
      if 'refexp' in pred_sample:
        print ('  ' + pred_sample['refexp'])
      else:
        refexp_tmp = self.refexp_dataset.loadRefexps(ids=pred_sample['refexp_id'])[0]
        print ('  ' + refexp_tmp['raw'])
    
    I = misc.imread(os.path.join(coco_image_dir, (img_coco['file_name'])))
    ax = plt.imshow(I)
    ax = plt.axis('off')
    ax = plt.title('IoU: %.3f, green bbox: GT, red bbox: predicted' % iou)
    cu.draw_bbox(plt.gca(), ann['bbox'], edge_color='green')
    cu.draw_bbox(plt.gca(), bbox_pred_top, edge_color='red')
    
    
class RefexpEvalGeneration(object):
  def __init__(self, refexp_dataset_path, coco_data_path):
    """Constructor for GoogleRefexpEvalGeneration class for evaluation.
    
    Args:
      refexp_dataset_path: path for the Google Refexp dataset file
    """
    # handle refexp dataset file
    assert refexp_dataset_path, "Refexp dataset file missing!"
    self.refexp_dataset_path = refexp_dataset_path
    print ('Loading Google Refexp dataset file for the generation task.')
    self.refexp_dataset = Refexp(refexp_dataset_path, coco_data_path) # Need to check - change
    self.gt_ann_ids_set = frozenset(self.refexp_dataset.getAnnIds()) # Need to check - change

