import os
import numpy as np
import torch
from PIL import Image

#####################################
# Class that takes the input instance masks
# and extracts bounding boxes on the fly
#####################################
class AppleDataset_mrcnn(object):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms

        # Load all image and mask files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)     # Each color of mask corresponds to a different instance with 0 being the background

        # Convert the PIL image to np array
        mask = np.array(mask)
        obj_ids = np.unique(mask)

        # Remove background id
        obj_ids = obj_ids[1:]

        # Split the color-encoded masks into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # Get bbox coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        h, w = mask.shape
        for ii in range(num_objs):
            pos = np.where(masks[ii])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax or ymin == ymax:
                continue

            xmin = np.clip(xmin, a_min=0, a_max=w)
            xmax = np.clip(xmax, a_min=0, a_max=w)
            ymin = np.clip(ymin, a_min=0, a_max=h)
            ymax = np.clip(ymax, a_min=0, a_max=h)
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # There is only one class (apples)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # All instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]


class AppleDataset_frcnn(object):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms

        # Load all image files (skip the masks folder)
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # No need to load masks for Faster-RCNN, so we skip the mask loading part
        # For Faster-RCNN, we just need the bounding box coordinates
        # Here, you should have bounding boxes (or you could use a simplified dataset without them)
        # If your dataset has ground-truth bounding boxes, return them here.

        # For simplicity, let's assume you have some dummy box generation or have a pre-defined function:
        boxes = np.array([[50, 50, 200, 200]])  # Dummy bounding box, replace with actual if needed
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels for each bounding box (in this case, just '1' for apple, but you can modify this)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        # Convert to the expected format for Faster-RCNN
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]
