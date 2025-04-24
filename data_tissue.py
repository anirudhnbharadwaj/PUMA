import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import logging
from shapely.geometry import shape

def generate_tissue_classes(tissue_root):
    tissue_classes = {
        'tissue_white_background': 0,
        'tissue_stroma': 1,
        'tissue_blood_vessel': 2,
        'tissue_tumor': 3,
        'tissue_epidermis': 4,
        'tissue_necrosis': 5
    }
    
    unique_class_dict = {}
    for geojson_file in os.listdir(tissue_root):
        if not geojson_file.endswith('.geojson'):
            continue
        with open(os.path.join(tissue_root, geojson_file), 'r') as f:
            geojson_data = json.load(f)
        for feature in geojson_data['features']:
            props = feature.get('properties', {})
            classification = props.get('classification', {})
            name = classification.get('name', 'unknown')
            if name in tissue_classes and name not in unique_class_dict:
                unique_class_dict[name] = tissue_classes[name]
    
    for name, class_id in unique_class_dict.items():
        logging.info(f"Class {class_id}: {name}")
    return tissue_classes

def create_mask_from_geojson(geojson_data, classes, shapes):
    class_mask = np.zeros(shapes, dtype=np.uint8)
    class_counts = {i: 0 for i in range(len(classes))}
    
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        classification = props.get('classification', {})
        name = classification.get('name', 'unknown')
        if name not in classes:
            continue
        class_id = classes[name]
        class_counts[class_id] += 1
        
        geom = shape(feature['geometry'])
        coordinates = []
        if geom.geom_type == 'Polygon':
            coordinates = [np.array(geom.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)]
            cv2.fillPoly(class_mask, coordinates, class_id)
        elif geom.geom_type == 'MultiPolygon':
            coordinates = [np.array(poly.exterior.coords, dtype=np.int32).reshape(-1, 1, 2) for poly in geom.geoms]
            cv2.fillPoly(class_mask, coordinates, class_id)
    
    return class_mask, class_counts

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

class PUMADatasetTissue(Dataset):
    def __init__(self, root, tissue_root, patch_size=256, stride=256, mode='train', reduced_load=False, reduced_size=31):
        self.root = root
        self.tissue_root = tissue_root
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.reduced_load = reduced_load
        self.reduced_size = reduced_size
        
        self.valid_images = [f for f in os.listdir(root) if f.endswith('.tif')]
        if not self.valid_images:
            raise ValueError(f"No valid images found in {root}")
        
        if self.reduced_load:
            logging.info(f"Reduced load enabled: selecting {self.reduced_size} random images")
            selected_indices = np.random.choice(len(self.valid_images), reduced_size, replace=False)
            self.valid_images = [self.valid_images[i] for i in selected_indices]
            logging.info(f"Selected {len(self.valid_images)} images")
        
        self.tissue_classes = generate_tissue_classes(tissue_root)
        self.class_counts = {i: 0 for i in range(len(self.tissue_classes))}
        
        self.patch_coords = []
        self.patch_class_distributions = []  # Store class distribution for each patch
        
        for idx, img_name in enumerate(self.valid_images):
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Skipping invalid image: {img_path}")
                continue
            
            tissue_geojson_path = os.path.join(self.tissue_root, f"{img_name.replace('.tif', '')}_tissue.geojson")
            with open(tissue_geojson_path, 'r') as f:
                tissue_geojson = json.load(f)
            tissue_class_mask, class_counts = create_mask_from_geojson(
                tissue_geojson, self.tissue_classes, (1024, 1024)
            )
            for class_id, count in class_counts.items():
                self.class_counts[class_id] += count
            
            patches, coords = patchify_image(img, patch_size, stride)
            for coord in coords:
                self.patch_coords.append((idx, coord))
                # Compute class distribution for this patch
                patch_i, patch_j = coord
                tissue_patch = tissue_class_mask[patch_i:patch_i+patch_size, patch_j:patch_j+patch_size]
                patch_counts = np.bincount(tissue_patch.ravel(), minlength=len(self.tissue_classes))
                patch_dist = patch_counts / (patch_counts.sum() + 1e-6)  # Normalize to get distribution
                self.patch_class_distributions.append(patch_dist)
        
        if not self.patch_coords:
            raise ValueError("No valid patches found")
        
        # Calculate overall pixel-based class distribution
        pixel_counts = np.zeros(len(self.tissue_classes))
        for idx in range(len(self.patch_coords)):
            img_idx, (patch_i, patch_j) = self.patch_coords[idx]
            img_name = self.valid_images[img_idx]
            tissue_geojson_path = os.path.join(self.tissue_root, f"{img_name.replace('.tif', '')}_tissue.geojson")
            with open(tissue_geojson_path, 'r') as f:
                tissue_geojson = json.load(f)
            tissue_class_mask, _ = create_mask_from_geojson(
                tissue_geojson, self.tissue_classes, (1024, 1024)
            )
            tissue_patch = tissue_class_mask[patch_i:patch_i+patch_size, patch_j:patch_j+patch_size]
            patch_counts = np.bincount(tissue_patch.ravel(), minlength=len(self.tissue_classes))
            pixel_counts += patch_counts
        
        self.class_distribution = pixel_counts / pixel_counts.sum()
        logging.info(f"Pixel-based class distribution: {dict(zip(self.tissue_classes.values(), self.class_distribution))}")
        
        # Refined class weights calculation: Logarithmic scaling with cap
        self.class_weights = np.log(1.0 / (self.class_distribution + 1e-6)) + 1  # Logarithmic scaling
        scaling_factor = 3.0  # Increase influence of rare classes
        max_weight = 10.0  # Cap the maximum weight to prevent extreme values
        min_weight = 0.05
        for i in range(len(self.class_weights)):
            if self.class_distribution[i] < 0.05:  # Rare classes
                self.class_weights[i] *= scaling_factor
            self.class_weights[i] = min(max(self.class_weights[i], min_weight), max_weight)
        self.class_weights /= self.class_weights.sum()  # Normalize
        logging.info(f"Refined class weights for loss (logarithmic with cap): {self.class_weights.tolist()}")

        # Refined sampling weights to achieve equal effective class distribution
        self.sampling_weights = []
        target_distribution = 1.0 / len(self.tissue_classes)  # Target: 1/6 ≈ 0.1667 for each class
        rare_class_threshold = 0.05  # Classes with distribution < 0.05 are considered rare
        rare_classes = [i for i in range(len(self.class_distribution)) if self.class_distribution[i] < rare_class_threshold]
        inverse_class_freq = 1.0 / (self.class_distribution + 1e-6)  # Inverse frequency of each class
        min_sampling_weight = 0.1  # Ensure all patches have a chance of being sampled

        # Extreme boost for very rare classes (like class 0 and class 5)
        extreme_boost_factor = 10000.0  # For classes with very low or zero distribution
        rare_boost_factor = 100.0  # For other rare classes

        # Initialize sampling weights
        for patch_dist in self.patch_class_distributions:
            weight = min_sampling_weight
            for class_id in range(len(self.tissue_classes)):
                if patch_dist[class_id] > 0:
                    class_weight = inverse_class_freq[class_id] * patch_dist[class_id]
                    if class_id in rare_classes:
                        if self.class_distribution[class_id] < 1e-4 or self.class_distribution[class_id] == 0.0:
                            class_weight *= extreme_boost_factor  # Extreme boost for classes 0 and 5
                        else:
                            class_weight *= rare_boost_factor  # Regular boost for classes 2 and 4
                    weight += class_weight
            self.sampling_weights.append(max(weight, min_sampling_weight))

        # Iteratively adjust sampling weights to achieve equal distribution
        max_iterations = 20  # Increased iterations to ensure convergence
        tolerance = 0.01  # Acceptable deviation from target (0.1667 ± 0.01)
        for iteration in range(max_iterations):
            # Compute effective distribution with current weights
            num_samples = len(self.sampling_weights) * 10
            sampling_weights_tensor = torch.tensor(self.sampling_weights, dtype=torch.float64)
            sampled_indices = torch.multinomial(sampling_weights_tensor, num_samples, replacement=True)
            
            sampled_pixel_counts = np.zeros(len(self.tissue_classes))
            for idx in sampled_indices:
                patch_dist = self.patch_class_distributions[idx]
                patch_pixel_counts = patch_dist * (self.patch_size * self.patch_size)
                sampled_pixel_counts += patch_pixel_counts
            
            effective_dist = sampled_pixel_counts / (sampled_pixel_counts.sum() + 1e-6)
            
            # Compute adjustment factors based on the ratio of target to current distribution
            adjustment_factors = target_distribution / (effective_dist + 1e-6)
            
            # Apply adjustment to sampling weights
            for i in range(len(self.sampling_weights)):
                patch_dist = self.patch_class_distributions[i]
                for class_id in range(len(self.tissue_classes)):
                    if patch_dist[class_id] > 0:
                        self.sampling_weights[i] *= adjustment_factors[class_id]
                # Ensure weights stay above minimum
                self.sampling_weights[i] = max(self.sampling_weights[i], min_sampling_weight)

            # Log the effective distribution after adjustment
            logging.info(f"Iteration {iteration + 1} - Effective class distribution: {dict(zip(self.tissue_classes.values(), effective_dist))}")

            # Check if all classes are within tolerance
            balanced = all(abs(dist - target_distribution) <= tolerance for dist in effective_dist if dist > 0)
            if balanced:
                logging.info(f"Equal effective distribution achieved after {iteration + 1} iterations")
                break

        self.effective_class_distribution = effective_dist
        logging.info(f"Final effective class distribution after sampling: {dict(zip(self.tissue_classes.values(), self.effective_class_distribution))}")
        logging.info(f"Computed refined sampling weights for balanced classes. Total patches: {len(self.sampling_weights)}")

        # Warn if any class remains at 0.0
        zero_classes = [class_id for class_id, dist in enumerate(self.effective_class_distribution) if dist == 0.0]
        if zero_classes:
            logging.warning(f"Classes {zero_classes} have an effective distribution of 0.0. Consider checking the dataset for these classes or adjusting the boost factors.")

    def get_sampler(self, subset_indices=None):
        """Return a WeightedRandomSampler for balanced classes, optionally for a subset of indices."""
        if subset_indices is None:
            weights = self.sampling_weights
            num_samples = len(self.sampling_weights)
        else:
            weights = [self.sampling_weights[i] for i in subset_indices]
            num_samples = len(subset_indices)
        
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True
        )

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        img_idx, (patch_i, patch_j) = self.patch_coords[idx]
        img_name = self.valid_images[img_idx]
        img_path = os.path.join(self.root, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        tissue_class_mask = None
        tissue_geojson_path = os.path.join(self.tissue_root, f"{img_name.replace('.tif', '')}_tissue.geojson")
        try:
            with open(tissue_geojson_path, 'r') as f:
                tissue_geojson = json.load(f)
            tissue_class_mask, _ = create_mask_from_geojson(
                tissue_geojson, self.tissue_classes, (1024, 1024)
            )
        except Exception as e:
            logging.error(f"Failed to load tissue GeoJSON for {img_name}: {str(e)}")
            raise
        
        img_patch = img[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size]
        tissue_class_patch = tissue_class_mask[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size]
        
        # Convert to tensor (no augmentations)
        img_patch = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        
        if img_patch.shape[1:] != (self.patch_size, self.patch_size):
            img_patch = torch.nn.functional.pad(img_patch, (0, self.patch_size-img_patch.shape[2], 0, self.patch_size-img_patch.shape[1]))
        if tissue_class_patch.shape != (self.patch_size, self.patch_size):
            tissue_class_patch = np.pad(tissue_class_patch, ((0, self.patch_size-tissue_class_patch.shape[0]), 
                                                            (0, self.patch_size-tissue_class_patch.shape[1])), mode='constant')
        
        return {
            'image': img_patch,
            'tissue_class_mask': torch.from_numpy(tissue_class_patch).long(),
            'image_name': img_name
        }