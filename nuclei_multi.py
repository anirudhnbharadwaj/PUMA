import torch
import torch.nn as nn
import torchvision.models.detection as detection
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import numpy as np
import logging
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
import os

class Config:
    """Configuration class for Mask R-CNN model parameters."""
    def __init__(self):
        # RPN parameters
        self.rpn_anchor_sizes = ((4, 8, 16), (8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256))
        self.rpn_aspect_ratios = ((0.2, 1.0, 8.0),) * 5
        self.rpn_pre_nms_top_n_train = 5000
        self.rpn_pre_nms_top_n_test = 2000
        self.rpn_post_nms_top_n_train = 3000
        self.rpn_post_nms_top_n_test = 1000
        self.rpn_nms_thresh = 0.5
        self.rpn_fg_iou_thresh = 0.5
        self.rpn_bg_iou_thresh = 0.5
        
        # Model head parameters
        self.num_classes = 4  # Background (0), Tumor (1), TILs (2), Others (3)
        self.mask_predictor_in_channels = 256
        self.mask_predictor_dim_reduced = 1024
        
        # Transform parameters
        self.transform_mean = [0.485, 0.456, 0.406]
        self.transform_std = [0.229, 0.224, 0.225]
        
        # Inference parameters
        self.score_threshold = 0.5

class NucleiModelMulti(nn.Module):
    def __init__(self):
        super(NucleiModelMulti, self).__init__()
        self.config = Config()
        
        self.nuclei_maskrcnn = detection.maskrcnn_resnet50_fpn(
            weights=None,
            progress=False,
            rpn_anchor_generator=detection.rpn.AnchorGenerator(
                sizes=self.config.rpn_anchor_sizes,
                aspect_ratios=self.config.rpn_aspect_ratios
            ),
            rpn_pre_nms_top_n_train=self.config.rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=self.config.rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=self.config.rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=self.config.rpn_post_nms_top_n_test,
            rpn_nms_thresh=self.config.rpn_nms_thresh,
            rpn_fg_iou_thresh=self.config.rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=self.config.rpn_bg_iou_thresh
        )

        pretrained_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=False)
        num_sizes_per_level = len(self.config.rpn_anchor_sizes[0])
        num_aspect_ratios = len(self.config.rpn_aspect_ratios[0])
        num_anchors_per_location = num_sizes_per_level * num_aspect_ratios
        num_anchors_default = 3
        num_anchors_custom = num_anchors_per_location

        if num_anchors_custom != num_anchors_default:
            repeat_factor = num_anchors_custom // num_anchors_default
            cls_logits_weight = pretrained_weights['rpn.head.cls_logits.weight']
            cls_logits_bias = pretrained_weights['rpn.head.cls_logits.bias']
            pretrained_weights['rpn.head.cls_logits.weight'] = cls_logits_weight.repeat(repeat_factor, 1, 1, 1)
            pretrained_weights['rpn.head.cls_logits.bias'] = cls_logits_bias.repeat(repeat_factor)
            bbox_pred_weight = pretrained_weights['rpn.head.bbox_pred.weight']
            bbox_pred_bias = pretrained_weights['rpn.head.bbox_pred.bias']
            pretrained_weights['rpn.head.bbox_pred.weight'] = bbox_pred_weight.repeat(repeat_factor, 1, 1, 1)
            pretrained_weights['rpn.head.bbox_pred.bias'] = bbox_pred_bias.repeat(repeat_factor)

        self.nuclei_maskrcnn.load_state_dict(pretrained_weights, strict=False)
        logging.info("Loaded pre-trained weights with adjusted RPN head parameters")

        # Modify for custom number of classes
        in_features = self.nuclei_maskrcnn.roi_heads.box_predictor.cls_score.in_features
        self.nuclei_maskrcnn.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, self.config.num_classes)
        self.nuclei_maskrcnn.roi_heads.mask_predictor = detection.mask_rcnn.MaskRCNNPredictor(
            in_channels=self.config.mask_predictor_in_channels,
            dim_reduced=self.config.mask_predictor_dim_reduced,
            num_classes=self.config.num_classes
        )

        self.maskrcnn_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.config.transform_mean, std=self.config.transform_std)
        ])
        logging.info(f"NucleiModelMulti initialized with Mask R-CNN for {self.config.num_classes} classes")

    def freeze_backbone(self, freeze=False):
        for param in self.nuclei_maskrcnn.backbone.parameters():
            param.requires_grad = not freeze
        logging.info(f"Backbone {'frozen' if freeze else 'unfrozen'}")

    def forward(self, images, targets=None, output_dir=None, batch_idx=None):
        logging.debug(f"NucleiModelMulti input: images shape: {images.shape}, min/max: {images.min().item():.2f}/{images.max().item():.2f}")
        logging.debug(f"Training mode: {self.training}, Targets provided: {targets is not None}")
        
        maskrcnn_images = images
        logging.debug(f"Mask R-CNN input: min/max: {maskrcnn_images.min().item():.2f}/{maskrcnn_images.max().item():.2f}")
        
        nuclei_outputs = None
        nuclei_instance_mask = np.zeros((images.size(0), images.size(2), images.size(3)), dtype=np.int64)
        
        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided in training mode")
            logging.debug(f"Targets type: {type(targets)}, Keys: {list(targets[0].keys()) if isinstance(targets, list) and len(targets) > 0 else None}")
            nuclei_outputs = self.nuclei_maskrcnn(maskrcnn_images, targets)
            return nuclei_outputs, nuclei_instance_mask
        else:
            nuclei_outputs = self.nuclei_maskrcnn(maskrcnn_images)
            for i, output in enumerate(nuclei_outputs):
                masks = output.get('masks', torch.zeros((0, 1, images.size(2), images.size(3))).to(images.device))
                scores = output.get('scores', torch.zeros((0,)).to(images.device))
                labels = output.get('labels', torch.zeros((0,), dtype=torch.int64).to(images.device))
                logging.debug(f"Batch {i}: Scores: {scores.tolist()}")
                logging.debug(f"Batch {i}: Labels: {labels.tolist()}")
                valid = (scores > self.config.score_threshold)
                logging.debug(f"Batch {i}: Number of valid predictions: {valid.sum().item()}")
                
                if valid.any():
                    valid_indices = torch.where(valid)[0]
                    sorted_indices = valid_indices[torch.argsort(scores[valid], descending=True)]
                    instance_ids = {}  # Track instance IDs per class
                    for j in sorted_indices:
                        mask = masks[j][0].cpu().numpy() > 0.5
                        class_id = labels[j].item()
                        if class_id not in instance_ids:
                            instance_ids[class_id] = 1
                        else:
                            instance_ids[class_id] += 1
                        instance_id = instance_ids[class_id]
                        new_instance_mask = (mask & (nuclei_instance_mask[i] == 0))  # Avoid overwriting
                        if new_instance_mask.any():
                            nuclei_instance_mask[i][new_instance_mask] = class_id * 1000 + instance_id
                            logging.debug(f"Batch {i}, Instance {instance_id}, Class {class_id}: Assigned mask with value {class_id * 1000 + instance_id}")
                    
                    if output_dir is not None and batch_idx is not None and batch_idx % 1000 == 0 and i == 0:
                        plt.figure(figsize=(5, 5))
                        plt.imshow(nuclei_instance_mask[i] % 1000, cmap='jet')  # Show instance IDs
                        plt.title("Predicted Instance Mask")
                        plt.axis('off')
                        plt.savefig(os.path.join(output_dir, f"instance_mask_batch_{batch_idx}.png"))
                        plt.close()
    
        return nuclei_outputs, nuclei_instance_mask