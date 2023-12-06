import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

def iou_width_height(boxes1, boxes2):

    intersection = torch.min(boxes1[...,0], boxes2[...,0]) * torch.min(boxes1[...,1], boxes2[...,1])
    union = ( boxes1[...,0] * boxes1[..., 1] + boxes2[..., 0]* boxes2[..., 1] - intersection )

    return intersection/union

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    if box_format == "midpoint":

        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2

    if box_format == "corners":

        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_x2 = boxes_labels[..., 1:2]
        box2_y1 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    x2 = torch.min(box1_x2, box2_x2)
    y1 = torch.max(box1_y1, box2_y1)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    box1_area = abs((box1_x2-box1_x1)* (box1_y2-box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):

    assert type(bboxes) == list

    bboxes = [ box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x : x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box 
            for box in bboxes:
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format = box_format
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    average_precisions = []

    epsilon= 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)


        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes==0:
            continue

        for detection_idx, detection in enumerate(detections):
            
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou> best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:

                if amount_bboxes[detection[0]][best_gt_idx] ==0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0][best_gt_idx]] = 1

                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat((torch.tensor([1]), recalls))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) /  len(average_precisions)


