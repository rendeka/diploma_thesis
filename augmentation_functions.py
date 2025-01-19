import numpy as np

def get_mask(samp_w, samp_h, x0, y0, cut_w, cut_h):
    """Return mask for given cut."""
    
    # fill with ones
    Mask = np.ones((samp_w, samp_h), dtype=int)
            
    # fill cut with zeros
    Mask[:cut_w,:cut_h] = np.zeros((cut_w, cut_h), dtype=int)
    
    # shift
    return np.roll(np.roll(Mask, x0, axis=1), y0, axis=0)

def get_cutmix_samples(Samp1, Samp2, Lab1, Lab2):
    """Generate two new CutMix samples."""
    
    # sample shape
    samp_w, samp_h = Samp1.shape
    
    # generate bounding box
    x0, y0 = np.random.randint((samp_w - 1, samp_h - 1))
    cut_w, cut_h = np.random.randint((samp_w - 1, samp_h - 1))
    
    # get mask
    Mask = get_mask(samp_w, samp_h, x0, y0, cut_w, cut_h)

    # create new cutmix samples
    Aug1 = Mask * Samp1 + (1 - Mask) * Samp2
    Aug2 = Mask * Samp2 + (1 - Mask) * Samp1

    # create labels for the new samples
    samp_area = samp_w * samp_h
    cut_area  = cut_w * cut_h
    
    eta = cut_area / samp_area
    AugLab1 = (1. - eta) * Lab1 + eta * Lab2
    AugLab2 = (1. - eta) * Lab2 + eta * Lab1
    
    return Aug1, Aug2, AugLab1, AugLab2


def get_mixup_samples(Samp1, Samp2, Lab1, Lab2):
    """Generate two new MixUp samples."""

    weight = np.random.rand()

    Aug1 = weight * Samp1 + (1 - weight) * Samp2 
    Aug2 = weight * Samp2 + (1 - weight) * Samp1

    AugLab1 = weight * Lab1 + (1 - weight) * Lab2 
    AugLab2 = weight * Lab2 + (1 - weight) * Lab1
    
    return Aug1, Aug2, AugLab1, AugLab2

def rot_fm_state(X):
    """Random coherent rotation of FM configurations."""
    
    Sz   = np.cos(X * np.pi)
    Rand = np.cos(2. * np.pi * np.random.rand(len(X)))
    
    SzNew = np.transpose(Rand * Sz.T)
    
    return np.arccos(SzNew) / np.pi