import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    N_cadre = 120  # 120 frames per scene (Total: 360 frames = 12s video)
    thetas = np.linspace(-180, 180, N_cadre, endpoint=False) # Base rotation
    
    # --- Scene 1: "ROLLERCOASTER" (Waves) ---
    phis_1 = -30.0 + 15.0 * np.sin(np.linspace(0, 4 * np.pi, N_cadre))
    radii_1 = np.full(N_cadre, 4.0)
    poses_1 = torch.stack([pose_spherical(t, p, r) for t, p, r in zip(thetas, phis_1, radii_1)], 0)

    # --- Scene 2: "VERTIGO / DYNAMIC ZOOM" ---
    phis_2 = np.full(N_cadre, -30.0)
    radii_2 = 4.0 + 1.5 * np.cos(np.linspace(0, 2 * np.pi, N_cadre))
    poses_2 = torch.stack([pose_spherical(t, p, r) for t, p, r in zip(thetas, phis_2, radii_2)], 0)

    # --- Scene 3: ASCENDING SPIRAL (Drone) ---
    phis_3 = np.linspace(-10.0, -60.0, N_cadre) 
    radii_3 = np.linspace(5.0, 3.0, N_cadre) 
    poses_3 = torch.stack([pose_spherical(t, p, r) for t, p, r in zip(thetas, phis_3, radii_3)], 0)

    # --- FINAL ASSEMBLY: Join all 3 scenes into a single list ---
    render_poses = torch.cat([poses_1, poses_2, poses_3], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


