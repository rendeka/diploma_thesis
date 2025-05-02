import random
import torch
import torchvision.transforms.v2 as v2
from skyrmion_dataset import SKYRMION
from typing import Callable
from functools import partial

def is_none(augment):
    """nargs='+' causes that augment can be string (or None), but also a list when you specify multiple arguments for --augment"""
    if augment is None:
        return True
    elif isinstance(augment, list):
        if None in augment and "tailored" not in augment:
            return True
    return False
    
def choose_augmentation(labels, args):
    if is_none(args.augment):
        return lambda images, labels: (images, labels), labels
    
    # We are creating augmented images during training instead of creating larger dataset that already contains
    # augmented images. In this way, we can artifically increase the size of our dataset by increasing number
    # of epochs, because in each epoch new data are generated.
    
    augment_set = set(args.augment) if isinstance(args.augment, list) else set([args.augment])

    if "tailored" in augment_set:
        batch_aug = TailoredAug(args)
    
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

    def get_cutmix_mask(
        self,
        batch_size: int,
        height: int = SKYRMION.H,
        width: int = SKYRMION.W,
        shape: str = 'random',  # 'random', 'rectangle' or 'circle'
    ) -> torch.Tensor:
        """Generate a binary mask for CutMix with rectangle or circle shape."""
        x = torch.randint(width, (batch_size,))
        y = torch.randint(height, (batch_size,))
        cut_w = torch.randint(1, width, (batch_size,))
        cut_h = torch.randint(1, height, (batch_size,))

        masks = torch.ones((batch_size, height, width), dtype=torch.float32)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device),
            torch.arange(width, device=x.device),
            indexing="ij"
        )

        if shape == "random":
            shape = random.choice(["rectangle", "circle"])

        for i in range(batch_size):
            if shape == 'rectangle':
                x1, y1 = max(x[i] - cut_w[i] // 2, 0), max(y[i] - cut_h[i] // 2, 0)
                x2, y2 = min(x1 + cut_w[i], width), min(y1 + cut_h[i], height)
                masks[i, y1:y2, x1:x2] = 0

            elif shape == 'circle':
                radius = torch.minimum(cut_w[i], cut_h[i]) // 2
                dist = (xx - x[i])**2 + (yy - y[i])**2
                masks[i][dist <= radius**2] = 0

            else:
                raise ValueError(f"Unsupported shape: {shape}")

        return masks
    
    def cutmix(self, batch_1: torch.Tensor, batch_2: torch.Tensor, labels_1: torch.Tensor, labels_2: torch.Tensor) -> list[torch.Tensor, torch.Tensor]:
        """cutmix augmentation on two groups."""
        batch_size = batch_1.shape[0]

        mask = self.get_cutmix_mask(batch_size).to(batch_1.device).unsqueeze(1)
        
        mixed_batch_1 = mask * batch_1 + (1 - mask) * batch_2
        mixed_batch_2 = mask * batch_2 + (1 - mask) * batch_1
        
        lambda_values = mask.mean(dim=[2, 3])
        mixed_labels_1 = lambda_values * labels_1 + (1 - lambda_values) * labels_2
        mixed_labels_2 = lambda_values * labels_2 + (1 - lambda_values) * labels_1
        
        return torch.cat([mixed_batch_1, mixed_batch_2]), torch.cat([mixed_labels_1, mixed_labels_2])

    
    def mixup(self, batch_1: torch.Tensor, batch_2: torch.Tensor, labels_1: torch.Tensor, labels_2: torch.Tensor) -> list[torch.Tensor, torch.Tensor]:
        """mixup augmentation on two groups."""
        batch_size = batch_1.shape[0]
        lambda_values = torch.rand(batch_size, 1, 1, 1, device=batch_1.device)
        
        mixed_batch_1 = lambda_values * batch_1 + (1 - lambda_values) * batch_2
        mixed_batch_2 = lambda_values * batch_2 + (1 - lambda_values) * batch_1
        
        lambda_values_flat = lambda_values[:, :, 0, 0]
        mixed_labels_1 = lambda_values_flat * labels_1 + (1 - lambda_values_flat) * labels_2
        mixed_labels_2 = lambda_values_flat * labels_2 + (1 - lambda_values_flat) * labels_1
        
        return torch.cat([mixed_batch_1, mixed_batch_2]), torch.cat([mixed_labels_1, mixed_labels_2])
    
    def augment_stack(self, batch_1: torch.Tensor, batch_2: torch.Tensor, labels_1: torch.Tensor, labels_2: torch.Tensor, augment: Callable) -> list[torch.Tensor, torch.Tensor]:
        
        batch_size_1, batch_size_2 = batch_1.shape[0], batch_2.shape[0]
        batch_size = min(batch_size_1, batch_size_2)

        if batch_size == 0:
            raise ValueError("`batch_size` has to be greater than zero")

        mixed_batch, mixed_labels = [], []

        while len(mixed_batch) < self.args.batch_size:
            shuffle_idx_1 = torch.randperm(batch_size_1) 
            shuffle_idx_2 = torch.randperm(batch_size_2)

            batch_A = batch_1[shuffle_idx_1][:batch_size]
            batch_B = batch_2[shuffle_idx_2][:batch_size]

            labels_A = labels_1[shuffle_idx_1][:batch_size]
            labels_B = labels_2[shuffle_idx_2][:batch_size]

            new_mixed_batch, new_mixed_labels = augment(batch_A, batch_B, labels_A, labels_B)
            mixed_batch.extend(new_mixed_batch)
            mixed_labels.extend(new_mixed_labels)
        
        return torch.stack(mixed_batch), torch.stack(mixed_labels)
    
    def rotate_fm_state(self, images: torch.Tensor) -> torch.Tensor:
        """Applies a random coherent rotation to each image in a batch.
        
        Args:
            images (torch.Tensor): A tensor of shape (B, 1, H, W) representing a batch of spin configurations.
            
        Returns:
            torch.Tensor: Rotated spin configuration tensor of the same shape.
        """
        batch_size = images.shape[0]

        random_rotations = torch.cos(2. * torch.pi * torch.rand(batch_size, device=images.device))
        
        random_rotations = random_rotations.view(batch_size, 1, 1, 1)

        spin_z = torch.cos(images * torch.pi)
        rotated_spin_z = random_rotations * spin_z

        return torch.arccos(rotated_spin_z) / torch.pi
    

    def roll_and_rotate_images(self, images: torch.Tensor) -> torch.Tensor:
        """Randomly rolls each image in both axes and applies a random 90-degree rotation.
        
        Args:
            images (torch.Tensor): A tensor of shape (B, 1, H, W) representing a batch of images.
            
        Returns:
            torch.Tensor: Transformed batch with random rolling and 90-degree rotations.
        """
        _, _, height, width = images.shape

        shift_h = torch.randint(0, height, (1,), device=images.device)
        shift_w = torch.randint(0, width, (1,), device=images.device)

        images = torch.roll(images, shifts=(shift_h, shift_w), dims=(2, 3))

        rotation = torch.randint(0, 4, (1,), device=images.device).item()

        images = torch.rot90(images, k=rotation, dims=(2, 3))

        return images
        
class TailoredAug(AUGMENT):
    """This augmentation rotates fe configurations, creates cutmixes between fe and sk phases and createx cutmixes between sk and sp phases. 
    It also rolls and rantom 90 degree rotate the images."""
    def __init__(self, args):
        super().__init__()

        self.args = args

        if isinstance(args.augment, str):
            raise ValueError(
                f"For the tailored augmentation, args.augment should contain a list of strings specifing the tailored augmentations.\n\
                args.augment contains only {self.args.augment} instead.")

        n_aug = len(self.args.augment)

        if not (isinstance(self.args.augment, list) and n_aug == 5):
            raise ValueError("args.augment must be list with 5 string elements for tailored augmentations.")
        
        self.augment = self.args.augment[1:]

        if n_aug == 5: # for transitional augmentations
            self.fe_sk = set([aug for aug in self.augment[:2] if ( aug != "None") and (aug is not None)])
            self.sk_sp = set([aug for aug in self.augment[2:] if ( aug != "None") and (aug is not None)])

        self.fe, self.sk, self.sp = self.args.phase_augment

    def check_equal(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        return x.shape == y.shape and torch.equal(x, y)

    def get_augment(self, trans_set: set):
        if not trans_set:    
            return None
        elif len(trans_set) == 2:
            aug = [self.cutmix, self.mixup][random.randint(0, 1)]
        elif "cutmix" in trans_set:
            aug =  self.cutmix
        elif "mixup" in trans_set:
            aug = self.mixup
        else:
            raise ValueError(f"cutmix or mixup expected, but got: {trans_set}")
        
        return partial(self.augment_stack, augment=aug)

    def pair_augment(self, images: torch.Tensor, labels: torch.Tensor, idx_1: list, idx_2:list, augment: Callable):

        if augment is None:
            return images, labels
        
        images_1, labels_1 = images[idx_1], labels[idx_1]
        images_2, labels_2 = images[idx_2], labels[idx_2]

        if len(idx_1) == 0:
            return images_2, labels_2
        
        elif len(idx_2) == 0:
            return images_1, labels_1

        if self.check_equal(idx_1, idx_2): # 90-degree rotate one set of images when combiding the same images
            images_2 = torch.rot90(images_2, k=1, dims=(2, 3))

        new_images, new_labels = augment(images_1, images_2, labels_1, labels_2)

        return new_images, new_labels

    def forward(self, images, labels):
        """Applies CutMix to fe-sk samples and MixUp to sk-sp samples within a batch."""

        batch_size = images.shape[0]
        
        idx_fe = (labels[:, 0] == 1).nonzero(as_tuple=True)[0]
        idx_sk = (labels[:, 1] == 1).nonzero(as_tuple=True)[0]
        idx_sp = (labels[:, 2] == 1).nonzero(as_tuple=True)[0]

        images = self.roll_and_rotate_images(images)
        images[idx_fe] = self.rotate_fm_state(images[idx_fe])

        if any(self.args.phase_augment) and random.random() > 0.5: # in-phase augmentations
            images_fe, labels_fe = self.pair_augment(images, labels, idx_fe, idx_fe, augment=self.get_augment(self.fe))
            images_sk, labels_sk = self.pair_augment(images, labels, idx_sk, idx_sk, augment=self.get_augment(self.sk))
            images_sp, labels_sp = self.pair_augment(images, labels, idx_sp, idx_sp, augment=self.get_augment(self.sp))

            images_aug = torch.cat((images_fe, images_sk, images_sp), dim=0)
            labels_aug = torch.cat((labels_fe, labels_sk, labels_sp), dim=0)

        else: # phase-transition augmentations
            images_fe_sk, labels_fe_sk = self.pair_augment(images, labels, idx_fe, idx_sk, augment=self.get_augment(self.fe_sk))
            images_sk_sp, labels_sk_sp = self.pair_augment(images, labels, idx_sk, idx_sp, augment=self.get_augment(self.sk_sp))

            images_aug = torch.cat((images_fe_sk, images_sk_sp), dim=0)
            labels_aug = torch.cat((labels_fe_sk, labels_sk_sp), dim=0)

        batch_size_aug = images_aug.shape[0]

        if self.args.keep_batch_size:
            if batch_size_aug < batch_size:
                # add original data until we have `batch_size` samples
                extra_indices = torch.randperm(batch_size)[:(batch_size - batch_size_aug)]
                images_aug = torch.cat((images_aug, images[extra_indices]), dim=0)
                labels_aug = torch.cat((labels_aug, labels[extra_indices]), dim=0)
            else:
                # take only `batch_size` samples
                selected = torch.randperm(batch_size_aug)[:batch_size]
                images_aug = images_aug[selected]
                labels_aug = labels_aug[selected]

            final_indices = torch.randperm(batch_size)
            return images_aug[final_indices], labels_aug[final_indices]

        else:
            indices = torch.randperm(batch_size_aug)
            return images_aug[indices], labels_aug[indices]
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

##### SOME OLD IDEAS TODO: check and maybe implement some

# # Augmentations

        
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