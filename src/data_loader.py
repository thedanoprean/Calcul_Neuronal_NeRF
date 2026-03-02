import os
import json
import cv2
import numpy as np
import torch

def load_nerf_data(base_path, split='train'):
    """
    Incarca imaginile si matricile de transformare pentru NeRF.
    Conform regulamentului, asigura split-ul de date.
    """
    json_path = os.path.join(base_path, f'transforms_{split}.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Nu s-a gasit fisierul: {json_path}")

    with open(json_path, 'r') as f:
        meta = json.load(f)

    imgs = []
    poses = []
    
    print(f"Incarcare date pentru split-ul: {split}...")
    
    for frame in meta['frames']:
        # Reconstituim calea catre imagine
        fname = os.path.join(base_path, frame['file_path'] + '.png')
        
        # Citim imaginea (RGBA)
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
            
        # Convertim in float32 si normalizam [0, 1]
        img = (img / 255.).astype(np.float32)
        imgs.append(img)
        
        # Extragem matricea de transformare a camerei
        poses.append(np.array(frame['transform_matrix']))

    imgs = np.array(imgs)
    poses = np.array(poses).astype(np.float32)
    
    # Calculam parametrii intrinseci (H, W, focal)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    return imgs, poses, [H, W, focal]