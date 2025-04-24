# import torch
# import torch.nn as nn
# import logging
# from monai.networks.nets import SwinUNETR

# class TissueModel(nn.Module):
#     def __init__(self, img_size=(128, 128)):
#         super(TissueModel, self).__init__()
#         self.num_classes = 6  # tissue_white_background (0), stroma (1), blood_vessel (2), tumor (3), epidermis (4), necrosis (5)
        
#         # Deeper Swin UNEtR configuration
#         self.model = SwinUNETR(
#             spatial_dims=2,
#             in_channels=3,
#             out_channels=self.num_classes,
#             img_size=img_size,  # Required: input image size
#             feature_size=96,    # Increased from default 48 for deeper feature extraction
#             depths=[2, 2, 6, 2],  # Deeper architecture: [2, 2, 6, 2] for more layers
#             num_heads=[3, 6, 12, 24],  # Number of attention heads per stage
#             drop_rate=0.1,    # Dropout rate
#             dropout_path_rate=0.2,  # Increased for regularization in deeper model
#             use_checkpoint=False  # Set to True for memory efficiency if needed
#         )
        
#         logging.info(f"TissueModel initialized with Swin UNEtR for {self.num_classes} classes, img_size={img_size}, depths={[2, 2, 6, 2]}")

#     def forward(self, images):
#         logging.debug(f"TissueModel input: images shape: {images.shape}, min/max: {images.min().item():.2f}/{images.max().item():.2f}")
#         outputs = self.model(images)
#         return outputs
    
    
    

import torch
import torch.nn as nn
import logging
from monai.networks.nets import DynUNet

class TissueModel(nn.Module):
    def __init__(self):
        super(TissueModel, self).__init__()
        self.num_classes = 6  # tissue_white_background (0), stroma (1), blood_vessel (2), tumor (3), epidermis (4), necrosis (5)
        
        # DynUNet configuration with a deeper architecture and deep supervision
        self.model = DynUNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=self.num_classes,
            kernel_size=[3, 3, 3, 3, 3, 3],  # Kernel sizes for each layer
            strides=[1, 2, 2, 2, 2, 2],      # Strides for downsampling/upsampling
            filters=[64, 128, 256, 512, 1024, 2048],  # Deeper with more filters
            upsample_kernel_size=[2, 2, 2, 2, 2],  # Upsampling kernel sizes
            dropout=0.1,  # Dropout rate
            deep_supervision=True,  # Enable deep supervision
            deep_supr_num=2  # Number of deep supervision heads (should output 3 tensors: main + 2 auxiliary)
        )
        
        logging.info(f"TissueModel initialized with DynUNet for {self.num_classes} classes, filters={self.model.filters}, deep_supervision=True")

    def forward(self, images):
        logging.debug(f"TissueModel input: images shape: {images.shape}, min/max: {images.min().item():.2f}/{images.max().item():.2f}")
        outputs = self.model(images)
        
        # DynUNet with deep_supervision=True returns a tuple of tensors
        # Each tensor should have shape [batch_size, num_classes, height, width]
        # However, if the output is not as expected, we need to handle it
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 5:
            # If output is [batch_size, num_deep_supervision, num_classes, height, width], split it
            logging.debug(f"DynUNet output is a single tensor with shape: {outputs.shape}")
            outputs = torch.split(outputs, split_size_or_sections=1, dim=1)  # Split along the deep supervision dim
            outputs = [output.squeeze(1) for output in outputs]  # Remove the extra dim
        elif not isinstance(outputs, (list, tuple)):
            outputs = [outputs]  # Wrap single tensor in a list
        
        # Log the output shapes
        for i, output in enumerate(outputs):
            logging.debug(f"TissueModel output {i}: shape: {output.shape}, min/max: {output.min().item():.2f}/{output.max().item():.2f}")
        
        return outputs
