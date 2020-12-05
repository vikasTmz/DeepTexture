'''
Reproject and transform object giving depth image
'''
import numpy as np
import pdb
import open3d as o3d
from scipy.spatial.transform import Rotation as R_
import scipy.linalg as linalg
import math
import cv2
import os
import random
import string

import torch

def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


def euler2rotmat(angle, axis='x'):
    if axis == 'x': axis_vec = [1, 0, 0]
    elif axis == 'y': axis_vec = [0, 1, 0],
    elif axis == 'z': axis_vec = [0, 0, 1],

    radians = math.radians(angle)
    rot_mat = rotate_mat(axis_vec, radians)

    return rot_mat

def depth_to_point(depth, cam_K, cam_W, save_dir):
        N = 256
        M = 256
        # create pixel location tensor
        xx = np.arange(0, 256, 1)
        yy = np.arange(0, 256, 1)
        py, px = np.meshgrid(xx, yy)
        p = np.stack((px,256-py), axis=0)
        p = p.reshape(2, N*M)

        mask_idx = np.where(depth.reshape(-1) != float('inf'))

        depth[depth==float('inf')] = 0
        d = depth.reshape(1, N*M)

        cx = 128.0
        cy = 128.0
        fx = 239.99998474
        fy = 239.99998474
        yy = d*(p[1] - cy) / fx
        xx = d*(p[0] - cx) / fy
        # create terms of mapping equation x = P^-1 * d*(qp - b)
        P = cam_K[:2, :2] 
        q = cam_K[2:3, 2:3]   
        b = cam_K[:2, 2:3].repeat(N*M, axis=1)
        Inv_P = np.linalg.inv(P)
        rightside = (p * q - b) * d
        x_xy = np.matmul(Inv_P, rightside)

        #x_xy = np.stack((xx,yy), axis=0)[:,0,:]
        x_xy = x_xy[:,mask_idx[0]]
        zz = d[0, mask_idx[0]]
        pcd_world = np.stack((-x_xy[0],x_xy[1],-zz),axis=1)

        pcd_world = pcd_world + cam_W[:,3]
        rot_z = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        pcd_world = np.matmul(pcd_world, rot_z)

        cam_W[:,3] = [0,0,0]

        f = open(save_dir+'/reproject_pcd_world.obj','w')
        for vert in pcd_world:
            f.write('v ' + str(vert[0]) + ' ' + str(vert[1]) + ' ' + str(vert[2]) + '\n')
        f.close()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_world)
        # print(np.asarray(pcd.points))

        o3d.io.write_point_cloud(os.path.join(save_dir, "reproject_pcd_world.ply"), pcd)

def depth_map_to_3d_torch(depth, cam_K, cam_W):
    """Derive 3D locations of each pixel of a depth map.

    Args:
        depth (torch.FloatTensor): tensor of size B x 1 x N x M
            with depth at every pixel
        cam_K (torch.FloatTensor): tensor of size B x 3 x 4 representing
            camera matrices
        cam_W (torch.FloatTensor): tensor of size B x 3 x 4 representing
            world matrices
    Returns:
        loc3d (torch.FloatTensor): tensor of size B x 3 x N x M
            representing color at given 3d locations
        mask (torch.FloatTensor):  tensor of size B x 1 x N x M with
            a binary mask if the given pixel is present or not
    """
    depth = torch.from_numpy(depth)
    cam_K = torch.from_numpy(cam_K)
    cam_W = torch.from_numpy(cam_W)

    N, M = depth.size()
    device = depth.device
    # Turn depth around. This also avoids problems with inplace operations
    depth = -depth.permute(1,0)

    zero_one_row = torch.tensor([[0., 0., 0., 1.]])
    zero_one_row = zero_one_row.expand(1, 4).to(device)
    # add row to world mat
    cam_W = torch.cat((cam_W, zero_one_row), dim=0)

    # clean depth image for mask
    # upperlimit =  1.e+10
    upperlimit = float("Inf")
    mask = (depth.abs() != upperlimit).float()
    depth[depth == upperlimit] = 0
    depth[depth == -1*upperlimit] = 0

    # 4d array to 2d array k=N*M
    d = depth.reshape(1,N * M)

    # create pixel location tensor
    px, py = torch.meshgrid([torch.arange(0, N), torch.arange(0, M)])
    px, py = px.to(device), py.to(device)

    p = torch.cat((
        px.expand(px.size(0), px.size(1)), 
        (M - py).expand(py.size(0), py.size(1))
    ), dim=0)
    p = p.reshape(2, py.size(0) * py.size(1))
    p = (p.float() / M * 2)
    
    # create terms of mapping equation x = P^-1 * d*(qp - b)
    P = cam_K[:2, :2].float().to(device)    
    q = cam_K[2:3, 2:3].float().to(device)   
    b = cam_K[:2, 2:3].expand(2, d.size(1)).to(device)
    Inv_P = torch.inverse(P).to(device)   

    rightside = (p.float() * q.float() - b.float()) * d.float()
    x_xy = torch.matmul(Inv_P, rightside)
    
    # add depth and ones to location in world coord system
    x_world = torch.cat((x_xy, d, torch.ones_like(d)), dim=0)

    # derive loactoion in object coord via loc3d = W^-1 * x_world
    Inv_W = torch.inverse(cam_W)
    Inv_W_exp = Inv_W.expand(4, 4)
    loc3d = torch.matmul(Inv_W_exp, x_world.double())
    loc3d = loc3d.reshape(4, N, M)

    loc3d = loc3d[:3,:,:].to(device)
    mask = mask.to(device)
    loc3d = loc3d.view(3, N * M)
    return loc3d, mask


def old_depth23d():
    # Inv_W = np.linalg.inv(cam_W)

    R = cam_W[0:3,0:3]
    t = cam_W[0:3,3]
    r = R_.from_matrix(R)

    # compute transformation matrix (R, t)
    # x_angle, z_angle, y_angle = r.as_euler('xyz', degrees=True)  # xyz ->xzy
    # rot_mat_x = euler2rotmat(90-x_angle, axis='x')
    # rot_mat_z = euler2rotmat(z_angle, axis='z')
    # rot_mat_y = euler2rotmat(-y_angle, axis='y')
    # dist = np.sqrt(np.square(t[0]) + np.square(t[1]) + np.square(t[2]))
    # R = np.dot(rot_mat_x, rot_mat_y)
    # t = np.array([0,0,dist])
    # t = t.reshape(3, 1)
    # cam_W = np.concatenate((R, t), axis=1)

    # save_dir = PATH
    # if not os.path.exists(save_dir): os.makedirs(save_dir)

    # depth_to_point(depth,cam_K, cam_W, save_dir)

PATH = "../../test_data/34080e679c1ae08aca92a4cdad802b45"


# filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
filename = "depth_3d"
f = open(filename + '.obj','w')

for j in range(0,10):
    depth = cv2.imread(PATH + '/depth/00' + str(j) + '.exr', -1)
    rgb = cv2.imread(PATH + '/image/00' + str(j) + '.png')
    size = rgb.shape[0]
    depth = np.array(depth[:,:,0])

    # cam_k and cam_W both obationed from blender
    camera_param = np.load(PATH + '/depth/cameras.npz')

    cam_K = camera_param["camera_mat_"+str(j)]
    cam_W = camera_param["world_mat_"+str(j)]

    loc3d, mask = depth_map_to_3d_torch(depth, cam_K, cam_W)
    loc3d_numpy = loc3d.cpu().detach().numpy() 

    for i in range(0, loc3d.shape[-1]):
        f.write('v ' + str(loc3d_numpy[0,i]) + ' ' + str(loc3d_numpy[1,i]) + ' ' + str(loc3d_numpy[2,i]) + \
         ' ' + str(rgb[i%size,i//size,2]) + ' ' + str(rgb[i%size,i//size,1]) + ' ' + str(rgb[i%size,i//size,0]) + '\n')


f.close()
