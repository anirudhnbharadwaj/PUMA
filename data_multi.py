# import os
# import json
# import numpy as np
# import cv2
# import torch
# from torch.utils.data import Dataset, WeightedRandomSampler
# import logging
# import torchvision.transforms as T
# from shapely.geometry import shape
# import matplotlib.pyplot as plt
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import monai.transforms as monaiT

# def generate_nuclei_classes(nuclei_root):
#     nuclei_classes = {'background': 0}
#     class_mapping = {
#         'nuclei_tumor': 1,  # Tumor
#         'nuclei_lymphocyte': 2,  # TILs
#         'nuclei_plasma_cell': 2,  # TILs
#         'nuclei_apoptosis': 3,  # Others
#         'nuclei_endothelium': 3,  # Others
#         'nuclei_stroma': 3,  # Others
#         'nuclei_histiocyte': 3,  # Others
#         'nuclei_melanophage': 3,  # Others
#         'nuclei_neutrophil': 3,  # Others
#         'nuclei_epithelium': 3  # Others
#     }
    
#     unique_class_dict = {'background': 0}
#     for geojson_file in os.listdir(nuclei_root):
#         if not geojson_file.endswith('.geojson'):
#             continue
#         with open(os.path.join(nuclei_root, geojson_file), 'r') as f:
#             geojson_data = json.load(f)
#         for feature in geojson_data['features']:
#             props = feature.get('properties', {})
#             classification = props.get('classification', {})
#             name = classification.get('name', 'unknown')
#             if name in class_mapping:
#                 nuclei_classes[name] = class_mapping[name]
#                 if name not in unique_class_dict:
#                     unique_class_dict[name] = class_mapping[name]
    
#     for name, class_id in unique_class_dict.items():
#         logging.info(f"Class {class_id}: {name}")
#     return nuclei_classes, class_mapping

# def create_mask_from_geojson(geojson_data, classes, class_mapping, shapes, instance=False):
#     class_mask = np.zeros(shapes, dtype=np.uint8)
#     instance_mask = np.zeros(shapes, dtype=np.int32)
#     instance_id = 1
#     nuclei_metrics = {
#         'areas': [],
#         'aspect_ratios': [],
#         'class_counts': {0: 0, 1: 0, 2: 0, 3: 0}
#     }
    
#     for feature in geojson_data['features']:
#         props = feature.get('properties', {})
#         classification = props.get('classification', {})
#         name = classification.get('name', 'unknown')
#         if name not in classes:
#             continue
#         class_id = classes[name]
#         nuclei_metrics['class_counts'][class_id] += 1
        
#         geom = shape(feature['geometry'])
#         temp_mask = np.zeros(shapes, dtype=np.uint8)
        
#         coordinates = []
#         if geom.geom_type == 'Polygon':
#             coordinates = [np.array(geom.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)]
#             cv2.fillPoly(temp_mask, coordinates, 1)
#             cv2.fillPoly(class_mask, coordinates, class_id)
#         elif geom.geom_type == 'MultiPolygon':
#             coordinates = [np.array(poly.exterior.coords, dtype=np.int32).reshape(-1, 1, 2) for poly in geom.geoms]
#             cv2.fillPoly(temp_mask, coordinates, 1)
#             cv2.fillPoly(class_mask, coordinates, class_id)
        
#         if instance:
#             instance_mask[temp_mask > 0] = instance_id
#             area = np.sum(temp_mask)
#             nuclei_metrics['areas'].append(area)
#             if coordinates:
#                 all_points = np.concatenate(coordinates, axis=0).reshape(-1, 2)
#                 x_min, y_min = np.min(all_points, axis=0)
#                 x_max, y_max = np.max(all_points, axis=0)
#                 width = x_max - x_min
#                 height = y_max - y_min
#                 aspect_ratio = width / height if height > 0 else 1.0
#                 nuclei_metrics['aspect_ratios'].append(aspect_ratio)
#             instance_id += 1
    
#     if nuclei_metrics['areas']:
#         logging.debug(f"Nuclei Metrics: "
#                      f"Smallest Area: {min(nuclei_metrics['areas']):.2f} pixels, "
#                      f"Largest Area: {max(nuclei_metrics['areas']):.2f} pixels, "
#                      f"Smallest Aspect Ratio: {min(nuclei_metrics['aspect_ratios']):.2f}, "
#                      f"Largest Aspect Ratio: {max(nuclei_metrics['aspect_ratios']):.2f}, "
#                      f"Class Counts: {nuclei_metrics['class_counts']}")
    
#     return class_mask, instance_mask, nuclei_metrics

# def patchify_image(image, patch_size, stride):
#     h, w = image.shape[:2]
#     patches = []
#     coords = []
#     for i in range(0, h - patch_size + 1, stride):
#         for j in range(0, w - patch_size + 1, stride):
#             patch = image[i:i+patch_size, j:j+patch_size]
#             patches.append(patch)
#             coords.append((i, j))
#     return patches, coords

# # Dynamic mean and std calculation
# def calculate_mean_std(dataset):
#     mean = np.zeros(3)
#     std = np.zeros(3)
#     for sample in dataset:
#         img = sample['image'].numpy().transpose(1, 2, 0)  # Convert to HWC
#         mean += img.mean(axis=(0, 1))
#         std += img.std(axis=(0, 1))
#     mean /= len(dataset)
#     std /= len(dataset)
#     logging.info(f"Dynamically calculated mean: {mean}, std: {std}")
#     return mean, std

# # Histopathology-specific augmentations
# def get_augmentation(mean=None, std=None):
#     # Geometric transforms for both image and masks
#     geometric_transforms = [
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.ShiftScaleRotate(
#             shift_limit=0.1,
#             scale_limit=0.1,
#             rotate_limit=45,
#             p=0.5,
#             border_mode=cv2.BORDER_CONSTANT,
#             interpolation=cv2.INTER_NEAREST
#         ),
#         A.ElasticTransform(
#             alpha=120,
#             sigma=120 * 0.05,
#             alpha_affine=120 * 0.03,
#             interpolation=cv2.INTER_NEAREST,
#             p=0.3
#         )
#     ]
    
#     if mean is None or std is None:
#         # Initial pipeline for mean/std calculation
#         return A.Compose(
#             geometric_transforms + [
#                 A.GaussianBlur(blur_limit=(3, 7), p=0.3),
#                 ToTensorV2()
#             ],
#             additional_targets={
#                 'class_mask': 'mask',
#                 'instance_mask': 'mask'
#             }
#         )
    
#     # Final pipeline
#     return A.Compose(
#         geometric_transforms + [
#             A.GaussianBlur(blur_limit=(3, 7), p=0.3),
#             A.Normalize(mean=mean, std=std),
#             ToTensorV2()
#         ],
#         additional_targets={
#             'class_mask': 'mask',
#             'instance_mask': 'mask'
#         }
#     )

# def get_mask_augmentation():
#     return A.Compose(
#         [
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.ShiftScaleRotate(
#                 shift_limit=0.1,
#                 scale_limit=0.1,
#                 rotate_limit=45,
#                 p=0.5,
#                 border_mode=cv2.BORDER_CONSTANT,
#                 interpolation=cv2.INTER_NEAREST
#             ),
#             A.ElasticTransform(
#                 alpha=120,
#                 sigma=120 * 0.05,
#                 alpha_affine=120 * 0.03,
#                 interpolation=cv2.INTER_NEAREST,
#                 p=0.3
#             )
#         ]
#     )

# class PUMADatasetMulti(Dataset):
#     def __init__(self, root, nuclei_root, patch_size=256, stride=256, mode='train', load_nuclei=True, maskrcnn_transforms=None, reduced_load=False, reduced_size=31, oversample_classes=False):
#         self.root = root
#         self.nuclei_root = nuclei_root
#         self.patch_size = patch_size
#         self.stride = stride
#         self.mode = mode
#         self.load_nuclei = load_nuclei
#         self.maskrcnn_transforms = maskrcnn_transforms
#         self.reduced_load = reduced_load
#         self.reduced_size = reduced_size
#         self.oversample_classes = oversample_classes
        
#         self.valid_images = [f for f in os.listdir(root) if f.endswith('.tif')]
#         if not self.valid_images:
#             raise ValueError(f"No valid images found in {root}")
        
#         if self.reduced_load:
#             logging.info(f"Reduced load enabled: selecting {self.reduced_size} random images")
#             selected_indices = np.random.choice(len(self.valid_images), reduced_size, replace=False)
#             self.valid_images = [self.valid_images[i] for i in selected_indices]
#             logging.info(f"Selected {len(self.valid_images)} images")
        
#         self.nuclei_classes, self.class_mapping = generate_nuclei_classes(nuclei_root) if self.load_nuclei else ({'background': 0}, {})
#         logging.info(f"Nuclei classes defined: {list(set(self.nuclei_classes.values()))}")
        
#         self.patch_coords = []
#         self.class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Patch-based counts
#         self.patch_class_weights = []  # Store pixel proportions per patch
#         for idx, img_name in enumerate(self.valid_images):
#             img_path = os.path.join(root, img_name)
#             img = cv2.imread(img_path)
#             if img is None:
#                 logging.warning(f"Skipping invalid image: {img_path}")
#                 continue
            
#             nuclei_class_mask = None
#             nuclei_instance_mask = None
#             if self.load_nuclei:
#                 nuclei_geojson_path = os.path.join(self.nuclei_root, f"{img_name.replace('.tif', '')}_nuclei.geojson")
#                 with open(nuclei_geojson_path, 'r') as f:
#                     nuclei_geojson = json.load(f)
#                 nuclei_class_mask, nuclei_instance_mask, _ = create_mask_from_geojson(
#                     nuclei_geojson, self.nuclei_classes, self.class_mapping, (1024, 1024), instance=True
#                 )
            
#             patches, coords = patchify_image(img, patch_size, stride)
#             for coord in coords:
#                 patch_i, patch_j = coord
#                 if self.load_nuclei:
#                     nuclei_class_patch = nuclei_class_mask[patch_i:patch_i+patch_size, patch_j:patch_j+patch_size]
#                     # Calculate pixel proportions for oversampling weights
#                     patch_pixel_counts = np.bincount(nuclei_class_patch.ravel(), minlength=4)
#                     total_pixels = patch_pixel_counts.sum()
#                     class_proportions = patch_pixel_counts / total_pixels if total_pixels > 0 else np.zeros(4)
#                     # Count patches with significant class presence (>5% pixels)
#                     for class_id in range(4):
#                         if class_proportions[class_id] > 0.05:
#                             self.class_counts[class_id] += 1
#                     self.patch_class_weights.append(class_proportions)
#                     self.patch_coords.append((idx, coord))
#                 else:
#                     self.class_counts[0] += 1
#                     self.patch_class_weights.append(np.array([1.0, 0.0, 0.0, 0.0]))  # Only background
#                     self.patch_coords.append((idx, coord))
        
#         if not self.patch_coords:
#             raise ValueError("No valid patches found")
        
#         logging.info(f"Initial patch-based class distribution: {self.class_counts}")
#         if self.oversample_classes:
#             # Mean target (smoother) instead of max
#             nuclei_counts = [self.class_counts[i] for i in [1, 2, 3]]
#             target_count = int(np.mean(nuclei_counts)) if nuclei_counts else 1000  # Fallback

#             weights = np.zeros(len(self.patch_coords))
#             for class_id in range(4):
#                 if self.class_counts[class_id] > 0:
#                     class_weight = target_count / self.class_counts[class_id]
#                     if class_id == 0:
#                         class_weight *= 2  # Background still gets doubled
#                     elif class_id == 1:
#                         class_weight = min(class_weight, 0.2)  # Cap Tumor to max 1x
#                 else:
#                     class_weight = 0.0

#                 for i, proportions in enumerate(self.patch_class_weights):
#                     weights[i] += class_weight * proportions[class_id]

#             # Now sample 4 × target_count patches
#             self.sampler = WeightedRandomSampler(weights, num_samples=target_count * 4, replacement=True)

#             # Sanity check final distribution
#             final_counts = {0: 0, 1: 0, 2: 0, 3: 0}
#             sampler_iter = iter(self.sampler)
#             for _ in range(target_count * 4):
#                 idx = next(sampler_iter) % len(self.patch_coords)
#                 nuclei_class_patch = self.get_patch_class_mask(idx)
#                 patch_pixel_counts = np.bincount(nuclei_class_patch.ravel(), minlength=4)
#                 for class_id in range(4):
#                     if patch_pixel_counts[class_id] / (self.patch_size * self.patch_size) > 0.05:
#                         final_counts[class_id] += 1

#             logging.info(f"[INFO] After mean+cap oversampling patch-based class distribution: {final_counts}")
#         else:
#             self.sampler = None
        
#         # Initialize augmentation with default (no normalization yet)
#         self.augmentation = get_augmentation()
#         self.mask_augmentation = get_mask_augmentation()
#         # Dynamic mean and std
#         # dummy_dataset = [self.__getitem__(i) for i in range(min(100, len(self)))]  # Sample 100 patches
#         self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#         # Update augmentation with computed mean and std
#         self.augmentation = get_augmentation(self.mean, self.std)

#     def get_patch_class_mask(self, idx):
#         img_idx, (patch_i, patch_j) = self.patch_coords[idx]
#         img_name = self.valid_images[img_idx]
#         nuclei_geojson_path = os.path.join(self.nuclei_root, f"{img_name.replace('.tif', '')}_nuclei.geojson")
#         with open(nuclei_geojson_path, 'r') as f:
#             nuclei_geojson = json.load(f)
#         nuclei_class_mask, _, _ = create_mask_from_geojson(
#             nuclei_geojson, self.nuclei_classes, self.class_mapping, (1024, 1024), instance=False
#         )
#         return nuclei_class_mask[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size]

#     def __len__(self):
#         return len(self.patch_coords) * 4 if self.oversample_classes else len(self.patch_coords)

#     def __getitem__(self, idx):
#         if self.oversample_classes:
#             idx = next(iter(self.sampler)) % len(self.patch_coords)
#         img_idx, (patch_i, patch_j) = self.patch_coords[idx]
#         img_name = self.valid_images[img_idx]
#         img_path = os.path.join(self.root, img_name)
        
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError(f"Failed to read image: {img_path}")
        
#         nuclei_class_mask = None
#         nuclei_instance_mask = None
#         if self.load_nuclei:
#             nuclei_geojson_path = os.path.join(self.nuclei_root, f"{img_name.replace('.tif', '')}_nuclei.geojson")
#             try:
#                 with open(nuclei_geojson_path, 'r') as f:
#                     nuclei_geojson = json.load(f)
#                 nuclei_class_mask, nuclei_instance_mask, _ = create_mask_from_geojson(
#                     nuclei_geojson, self.nuclei_classes, self.class_mapping, (1024, 1024), instance=True
#                 )
#             except Exception as e:
#                 logging.error(f"Failed to load nuclei GeoJSON for {img_name}: {str(e)}")
#                 raise
        
#         img_patch = img[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size]
#         nuclei_class_patch = nuclei_class_mask[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size] if self.load_nuclei else np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
#         nuclei_instance_patch = nuclei_instance_mask[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size] if self.load_nuclei else np.zeros((self.patch_size, self.patch_size), dtype=np.int32)
        
#         # Apply image augmentations
#         img_augmented = self.augmentation(image=img_patch)
#         img_patch = img_augmented['image']
        
#         # Apply mask augmentations separately
#         mask_augmented = self.mask_augmentation(
#             image=nuclei_class_patch,  # Dummy image to satisfy albumentations
#             mask=nuclei_class_patch,
#             instance_mask=nuclei_instance_patch
#         )
#         nuclei_class_patch = mask_augmented['mask']
#         nuclei_instance_patch = mask_augmented['instance_mask']
        
#         logging.debug(f"Patch {idx} Augmentation: Applied {img_augmented.get('replay', {}).get('transform', 'None')}")
        
#         # Ensure correct shape for image
#         if img_patch.shape[1:] != (self.patch_size, self.patch_size):
#             img_patch = torch.nn.functional.pad(
#                 img_patch,
#                 (0, self.patch_size - img_patch.shape[2], 0, self.patch_size - img_patch.shape[1]),
#                 mode='constant',
#                 value=0
#             )
        
#         # Ensure correct shape for masks
#         if nuclei_class_patch.shape != (self.patch_size, self.patch_size):
#             nuclei_class_patch = np.pad(
#                 nuclei_class_patch,
#                 ((0, self.patch_size - nuclei_class_patch.shape[0]), 
#                  (0, self.patch_size - nuclei_class_patch.shape[1])),
#                 mode='constant'
#             )
#             nuclei_instance_patch = np.pad(
#                 nuclei_instance_patch,
#                 ((0, self.patch_size - nuclei_instance_patch.shape[0]), 
#                  (0, self.patch_size - nuclei_instance_patch.shape[1])),
#                 mode='constant'
#             )
        
#         return {
#             'image': img_patch,
#             'nuclei_class_mask': torch.from_numpy(nuclei_class_patch).long(),
#             'nuclei_instance_mask': torch.from_numpy(nuclei_instance_patch).long(),
#             'image_name': img_name
#         }
        
        
        
        
        
        
        
        
import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import logging
import torchvision.transforms as T
from shapely.geometry import shape
import matplotlib.pyplot as plt

def generate_nuclei_classes(nuclei_root):
    nuclei_classes = {'background': 0}
    class_mapping = {
        'nuclei_tumor': 1,  # Tumor
        'nuclei_lymphocyte': 2,  # TILs
        'nuclei_plasma_cell': 2,  # TILs
        'nuclei_apoptosis': 3,  # Others
        'nuclei_endothelium': 3,  # Others
        'nuclei_stroma': 3,  # Others
        'nuclei_histiocyte': 3,  # Others
        'nuclei_melanophage': 3,  # Others
        'nuclei_neutrophil': 3,  # Others
        'nuclei_epithelium': 3  # Others
    }
    
    # Dictionary to store unique class names and their IDs
    unique_class_dict = {'background': 0}
    
    for geojson_file in os.listdir(nuclei_root):
        if not geojson_file.endswith('.geojson'):
            continue
        with open(os.path.join(nuclei_root, geojson_file), 'r') as f:
            geojson_data = json.load(f)
        for feature in geojson_data['features']:
            props = feature.get('properties', {})
            classification = props.get('classification', {})
            name = classification.get('name', 'unknown')
            if name in class_mapping:
                nuclei_classes[name] = class_mapping[name]
                
                # Add to unique_class_dict only if not already present
                if name not in unique_class_dict:
                    unique_class_dict[name] = class_mapping[name]
    
    # Log unique class names and IDs once
    for name, class_id in unique_class_dict.items():
        logging.info(f"Class {class_id}: {name}")
    
    return nuclei_classes, class_mapping

def create_mask_from_geojson(geojson_data, classes, class_mapping, shapes, instance=False):
    class_mask = np.zeros(shapes, dtype=np.uint8)
    instance_mask = np.zeros(shapes, dtype=np.int32)
    instance_id = 1
    nuclei_metrics = {
        'areas': [],
        'aspect_ratios': [],
        'class_counts': {0: 0, 1: 0, 2: 0, 3: 0}
    }
    
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        classification = props.get('classification', {})
        name = classification.get('name', 'unknown')
        if name not in classes:
            continue
        class_id = classes[name]
        nuclei_metrics['class_counts'][class_id] += 1
        
        geom = shape(feature['geometry'])
        temp_mask = np.zeros(shapes, dtype=np.uint8)
        
        coordinates = []
        if geom.geom_type == 'Polygon':
            coordinates = [np.array(geom.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)]
            cv2.fillPoly(temp_mask, coordinates, 1)
            cv2.fillPoly(class_mask, coordinates, class_id)
        elif geom.geom_type == 'MultiPolygon':
            coordinates = [np.array(poly.exterior.coords, dtype=np.int32).reshape(-1, 1, 2) for poly in geom.geoms]
            cv2.fillPoly(temp_mask, coordinates, 1)
            cv2.fillPoly(class_mask, coordinates, class_id)
        
        if instance:
            instance_mask[temp_mask > 0] = instance_id
            area = np.sum(temp_mask)  # Area in pixels
            nuclei_metrics['areas'].append(area)
            # Calculate bounding box for aspect ratio
            if coordinates:
                all_points = np.concatenate(coordinates, axis=0).reshape(-1, 2)
                x_min, y_min = np.min(all_points, axis=0)
                x_max, y_max = np.max(all_points, axis=0)
                width = x_max - x_min
                height = y_max - y_min
                aspect_ratio = width / height if height > 0 else 1.0
                nuclei_metrics['aspect_ratios'].append(aspect_ratio)
            instance_id += 1
    
    # Log nuclei metrics
    if nuclei_metrics['areas']:
        logging.debug(f"Nuclei Metrics: "
                     f"Smallest Area: {min(nuclei_metrics['areas']):.2f} pixels, "
                     f"Largest Area: {max(nuclei_metrics['areas']):.2f} pixels, "
                     f"Smallest Aspect Ratio: {min(nuclei_metrics['aspect_ratios']):.2f}, "
                     f"Largest Aspect Ratio: {max(nuclei_metrics['aspect_ratios']):.2f}, "
                     f"Class Counts: {nuclei_metrics['class_counts']}")
    
    return class_mask, instance_mask, nuclei_metrics

def patchify_image(image, patch_size, stride):
    h, w = image.shape[:2]
    patches = []
    coords = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            coords.append((i, j))
    return patches, coords

class PUMADatasetMulti(Dataset):
    def __init__(self, root, nuclei_root, patch_size=256, stride=256, mode='train', load_nuclei=True, maskrcnn_transforms=None, reduced_load=False, reduced_size=31):
        self.root = root
        self.nuclei_root = nuclei_root
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.load_nuclei = load_nuclei
        self.maskrcnn_transforms = maskrcnn_transforms
        self.reduced_load = reduced_load
        self.reduced_size = reduced_size
        
        # Get all valid images
        self.valid_images = [f for f in os.listdir(root) if f.endswith('.tif')]
        if not self.valid_images:
            raise ValueError(f"No valid images found in {root}")
        
        # Randomly select 41 images if reduced_load is True
        if self.reduced_load:
            logging.info(f"Reduced load enabled: selecting {self.reduced_size} random images")
            # np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(len(self.valid_images), reduced_size, replace=False)
            self.valid_images = [self.valid_images[i] for i in selected_indices]
            logging.info(f"Selected {len(self.valid_images)} images")
        
        self.nuclei_classes, self.class_mapping = generate_nuclei_classes(nuclei_root) if self.load_nuclei else ({'background': 0}, {})
        logging.info(f"Nuclei classes defined: {list(set(self.nuclei_classes.values()))}")
        
        self.patch_coords = []
        for idx, img_name in enumerate(self.valid_images):
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Skipping invalid image: {img_path}")
                continue
            
            nuclei_class_mask = None
            nuclei_instance_mask = None
            if self.load_nuclei:
                nuclei_geojson_path = os.path.join(self.nuclei_root, f"{img_name.replace('.tif', '')}_nuclei.geojson")
                with open(nuclei_geojson_path, 'r') as f:
                    nuclei_geojson = json.load(f)
                nuclei_class_mask, nuclei_instance_mask, _ = create_mask_from_geojson(
                    nuclei_geojson, self.nuclei_classes, self.class_mapping, (1024, 1024), instance=True
                )
            
            patches, coords = patchify_image(img, patch_size, stride)
            for coord in coords:
                patch_i, patch_j = coord
                if self.load_nuclei:
                    nuclei_class_patch = nuclei_class_mask[patch_i:patch_i+patch_size, patch_j:patch_j+patch_size]
                    nuclei_instance_patch = nuclei_instance_mask[patch_i:patch_i+patch_size, patch_j:patch_j+patch_size]
                    self.patch_coords.append((idx, coord))
                else:
                    self.patch_coords.append((idx, coord))
        
        if not self.patch_coords:
            raise ValueError("No valid patches found")
    
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        img_idx, (patch_i, patch_j) = self.patch_coords[idx]
        img_name = self.valid_images[img_idx]
        img_path = os.path.join(self.root, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        nuclei_class_mask = None
        nuclei_instance_mask = None
        if self.load_nuclei:
            nuclei_geojson_path = os.path.join(self.nuclei_root, f"{img_name.replace('.tif', '')}_nuclei.geojson")
            try:
                with open(nuclei_geojson_path, 'r') as f:
                    nuclei_geojson = json.load(f)
                nuclei_class_mask, nuclei_instance_mask, _ = create_mask_from_geojson(
                    nuclei_geojson, self.nuclei_classes, self.class_mapping, (1024, 1024), instance=True
                )
            except Exception as e:
                logging.error(f"Failed to load nuclei GeoJSON for {img_name}: {str(e)}")
                raise
        
        img_patch = img[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size]
        nuclei_class_patch = nuclei_class_mask[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size] if self.load_nuclei else np.zeros((self.patch_size, self.patch_size), dtype=np.int64)
        nuclei_instance_patch = nuclei_instance_mask[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size] if self.load_nuclei else np.zeros((self.patch_size, self.patch_size), dtype=np.int64)
        
        logging.debug(f"Patch {idx} ({img_name}, {patch_i}, {patch_j}): "
                     f"nuclei_class_patch values: {np.unique(nuclei_class_patch)}, "
                     f"nuclei_instance_patch values: {np.unique(nuclei_instance_patch)}")
        
        if img_patch.shape[:2] != (self.patch_size, self.patch_size):
            img_patch = np.pad(img_patch, ((0, self.patch_size-img_patch.shape[0]), 
                                         (0, self.patch_size-img_patch.shape[1]), (0, 0)), mode='constant')
        if self.load_nuclei and nuclei_class_patch.shape != (self.patch_size, self.patch_size):
            nuclei_class_patch = np.pad(nuclei_class_patch, ((0, self.patch_size-nuclei_class_patch.shape[0]), 
                                                           (0, self.patch_size-nuclei_class_patch.shape[1])), mode='constant')
            nuclei_instance_patch = np.pad(nuclei_instance_patch, ((0, self.patch_size-nuclei_instance_patch.shape[0]), 
                                                                 (0, self.patch_size-nuclei_instance_patch.shape[1])), mode='constant')
        
        if self.maskrcnn_transforms is not None:
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
            img_patch = self.maskrcnn_transforms(img_patch)
        else:
            img_patch = img_patch.transpose(2, 0, 1)
            img_patch = torch.from_numpy(img_patch).float()
        
        logging.debug(f"Patch {idx}: Image shape: {img_patch.shape}, min/max: {img_patch.min():.2f}/{img_patch.max():.2f}, "
                     f"Nuclei class mask shape: {nuclei_class_patch.shape}, min/max: {nuclei_class_patch.min()}/{nuclei_class_patch.max()}, "
                     f"Nuclei instance mask shape: {nuclei_instance_patch.shape}, min/max: {nuclei_instance_patch.min()}/{nuclei_instance_patch.max()}")
        
        return {
            'image': img_patch,
            'nuclei_class_mask': torch.from_numpy(nuclei_class_patch).long(),
            'nuclei_instance_mask': torch.from_numpy(nuclei_instance_patch).long(),
            'image_name': img_name
        }