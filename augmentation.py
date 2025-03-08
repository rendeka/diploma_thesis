import torch
import torchvision.transforms.v2 as v2
from skyrmion_dataset import SKYRMION

def is_none(augment):
    """nargs='+' causes that augment can be string (or None), but also a list when you specify multiple arguments for --augment"""
    if augment is None:
        return True
    elif isinstance(augment, list):
        if None in augment:
            return True
    return False
    
def choose_augmentation(labels, augment):
    if is_none(augment):
        return lambda images, labels: (images, labels), labels
    
    # We are creating augmented images during training instead of creating larger dataset that already contains
    # augmented images. In this way, we can artifically increase the size of our dataset by increasing number
    # of epochs, because in each epoch new data are generated.
    
    augment_set = set(augment) if isinstance(augment, list) else set([augment])

    if "custom" in augment_set:
        batch_aug = CustomAug()
    
    else:
        probs = labels.float().mean(dim=0)
        labels = torch.argmax(labels, dim=1) # v2 transformations need sparse representation

        v2_cutmix = v2.CutMix(num_classes=len(SKYRMION.LABELS))
        v2_mixup = v2.Compose([v2.ToDtype(torch.float32), v2.MixUp(num_classes=len(SKYRMION.LABELS))])
        
        if "adaptive" in augment_set:
            # This idea (with adaptive option) is based on the observation that cutmixes improves fe-sk transitions and mixups improve sk-sp transitions
            # if there are more fe examples than sp examples choose cutmix, else mixup.
            # Currently, we are NOT automatically increasing the number of epochs for the runs with augmentations
            if probs[0] > probs[2]:
                batch_aug = v2_cutmix
            else:
                batch_aug = v2_mixup

        elif {"cutmix", "mixup"}.issubset(augment_set):
            batch_aug = v2.RandomChoice([
                v2_cutmix,
                v2_mixup
            ])

        elif "cutmix" in augment_set:
            batch_aug = v2_cutmix
        else:
            batch_aug = v2_mixup

    return batch_aug, labels

class AUGMENT(v2.Transform):
    def __init__(self):
        super().__init__()

    def get_cutmix_mask(self, batch_size, height=SKYRMION.H, width=SKYRMION.W):
        """Generate a binary mask for CutMix."""
        x = torch.randint(width, (batch_size,))
        y = torch.randint(height, (batch_size,))
        cut_w = torch.randint(1, width, (batch_size,))
        cut_h = torch.randint(1, height, (batch_size,))
        
        masks = torch.ones((batch_size, height, width), dtype=torch.float32)
        for i in range(batch_size):
            x1, y1 = max(x[i] - cut_w[i] // 2, 0), max(y[i] - cut_h[i] // 2, 0)
            x2, y2 = min(x1 + cut_w[i], width), min(y1 + cut_h[i], height)
            masks[i, y1:y2, x1:x2] = 0
        
        return masks

    def cutmix(self, batch_1, batch_2, labels_1, labels_2):
        """cutmix augmentation on two groups."""
        batch_size = min(batch_1.shape[0], batch_2.shape[0])

        batch_1 = batch_1[:batch_size]
        batch_2 = batch_2[:batch_size]

        labels_1 = labels_1[:batch_size]
        labels_2 = labels_2[:batch_size]

        mask = self.get_cutmix_mask(batch_size).to(batch_1.device).unsqueeze(1)
        
        mixed_batch_1 = mask * batch_1 + (1 - mask) * batch_2
        mixed_batch_2 = mask * batch_2 + (1 - mask) * batch_1
        
        lambda_values = mask.mean(dim=[2, 3])
        mixed_labels_1 = lambda_values * labels_1 + (1 - lambda_values) * labels_2
        mixed_labels_2 = lambda_values * labels_2 + (1 - lambda_values) * labels_1
        
        return torch.cat([mixed_batch_1, mixed_batch_2]), torch.cat([mixed_labels_1, mixed_labels_2])
    
    def mixup(self, batch_1, batch_2, labels_1, labels_2):
        """mixup augmentation on two groups."""
        batch_size = min(batch_1.shape[0], batch_2.shape[0])

        batch_1 = batch_1[:batch_size]
        batch_2 = batch_2[:batch_size]

        labels_1 = labels_1[:batch_size]
        labels_2 = labels_2[:batch_size]
        
        lambda_values = torch.rand(batch_size, 1, 1, 1, device=batch_1.device)
        
        mixed_batch_1 = lambda_values * batch_1 + (1 - lambda_values) * batch_2
        mixed_batch_2 = lambda_values * batch_2 + (1 - lambda_values) * batch_1
        
        lambda_values_flat = lambda_values[:, :, 0, 0]
        mixed_labels_1 = lambda_values_flat * labels_1 + (1 - lambda_values_flat) * labels_2
        mixed_labels_2 = lambda_values_flat * labels_2 + (1 - lambda_values_flat) * labels_1
        
        return torch.cat([mixed_batch_1, mixed_batch_2]), torch.cat([mixed_labels_1, mixed_labels_2])
    
class CustomAug(AUGMENT):
    def __init__(self):
        super().__init__()

    def pair_augment(self, images, labels, idx_1, idx_2, augment):
        
        images_1, labels_1 = images[idx_1], labels[idx_1]
        images_2, labels_2 = images[idx_2], labels[idx_2]

        if len(idx_1) == 0:
            return images_2, labels_2
        elif len(idx_2) == 0:
            return images_1, labels_1

        new_images, new_labels = augment(images_1, images_2, labels_1, labels_2)

        return new_images, new_labels

    def forward(self, images, labels):
        """Applies CutMix to fe-sk samples and MixUp to sk-sp samples within a batch."""

        idx_fe = (labels[:, 0] == 1).nonzero(as_tuple=True)[0]
        idx_sk = (labels[:, 1] == 1).nonzero(as_tuple=True)[0]
        idx_sp = (labels[:, 2] == 1).nonzero(as_tuple=True)[0]

        images_fe_sk, labels_fe_sk = self.pair_augment(images, labels, idx_fe, idx_sk, augment=self.cutmix)
        images_sk_sp, labels_sk_sp = self.pair_augment(images, labels, idx_sk, idx_sp, augment=self.mixup)

        images = torch.cat((images_fe_sk, images_sk_sp), dim=0)
        labels = torch.cat((labels_fe_sk, labels_sk_sp), dim=0)

        idx_shuffle = torch.randperm(images.shape[0])
        return images[idx_shuffle], labels[idx_shuffle]
    
    def __call__(self, images, labels):
        return self.forward(images, labels)

##### SOME OLD IDEAS TODO: check and maybe implement some
# # Rotating an image by 90 degrees with the chance 1/2
# def rotate90(image):
#     if np.random.rand() < 0.5:
#         return v2.functional.rotate(image, angle=90)
#     else:
#         return image

# # Augmentations
# augmentations = []
# if args.augment:
#     augmentations.append(v2.RandomHorizontalFlip())
#     augmentations.append(rotate90)
        
#     # augmentations.append(v2.RandomCrop((SKYRMION.H, SKYRMION.W), padding=4, fill=127))
# # if "autoaugment" in args.augment:
# #     augmentations.append(v2.AutoAugment(v2.AutoAugmentPolicy.SKYRMION, fill=127))
# # if randaugment := re.search(r"randaugment-(\d+)-(\d+)", args.augment):
# #     n, m = map(int, randaugment.groups())
# #     augmentations.append(v2.RandAugment(n, m, fill=127))
# # if "cutout" in args.augment:
# #     def cutout(image):
# #         y, x = np.random.randint(SKYRMION.H), np.random.randint(SKYRMION.W)
# #         image = image.clone()
# #         image[:, max(0, y - 8):y + 8, max(0, x - 8):x + 8] = 127
# #         return image
# #     augmentations.append(v2.Lambda(cutout))
#     augmentations = v2.Compose(augmentations)
#     train = train.transform(lambda image, label: (augmentations(image.permute(2, 0, 1)).permute(1, 2, 0), label))