import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector
import random
import numpy as np

HEMI_SPHERE = [[0.0000, 0.8584, 3.0018],[0.0000, 1.6838, 2.7133],[0.0000, 2.4445, 2.2448],[0.0000, 3.1113, 1.6143],[0.0000, 3.6585, 0.8460],[0.0000, 4.0651, -0.0306],[0.0000, 4.3155, -0.9817],[0.1903, 0.8419, 3.0018],[0.3733, 1.6515, 2.7133],[0.5419, 2.3975, 2.2448],[0.6897, 3.0515, 1.6143],[0.8111, 3.5882, 0.8460],[0.9012, 3.9870, -0.0306],[0.9567, 4.2325, -0.9817],[0.3733, 0.7931, 3.0018],[0.7322, 1.5556, 2.7133],[1.0630, 2.2584, 2.2448],[1.3530, 2.8744, 1.6143],[1.5909, 3.3800, 0.8460],[1.7678, 3.7556, -0.0306],[1.8767, 3.9870, -0.9817],[0.5419, 0.7137, 3.0018],[1.0630, 1.4000, 2.7133],[1.5433, 2.0325, 2.2448],[1.9642, 2.5869, 1.6143],[2.3097, 3.0419, 0.8460],[2.5664, 3.3800, -0.0306],[2.7245, 3.5882, -0.9817],[0.6897, 0.6070, 3.0018],[1.3530, 1.1906, 2.7133],[1.9642, 1.7285, 2.2448],[2.5000, 2.2000, 1.6143],[2.9397, 2.5869, 0.8460],[3.2664, 2.8744, -0.0306],[3.4676, 3.0515, -0.9817],[0.8111, 0.4769, 3.0018],[1.5909, 0.9355, 2.7133],[2.3097, 1.3581, 2.2448],[2.9397, 1.7285, 1.6143],[3.4567, 2.0325, 0.8460],[3.8409, 2.2584, -0.0306],[4.0775, 2.3975, -0.9817],[0.9012, 0.3285, 3.0018],[1.7678, 0.6444, 2.7133],[2.5664, 0.9355, 2.2448],[3.2664, 1.1906, 1.6143],[3.8409, 1.4000, 0.8460],[4.2678, 1.5556, -0.0306],[4.5306, 1.6515, -0.9817],[0.9567, 0.1675, 3.0018],[1.8767, 0.3285, 2.7133],[2.7245, 0.4769, 2.2448],[3.4676, 0.6070, 1.6143],[4.0775, 0.7137, 0.8460],[4.5306, 0.7931, -0.0306],[4.8097, 0.8419, -0.9817],[0.9755, -0.0000, 3.0018],[1.9134, -0.0000, 2.7133],[2.7779, -0.0000, 2.2448],[3.5355, -0.0000, 1.6143],[4.1573, -0.0000, 0.8460],[4.6194, -0.0000, -0.0306],[4.9039, -0.0000, -0.9817],[0.9567, -0.1675, 3.0018],[1.8767, -0.3285, 2.7133],[2.7245, -0.4769, 2.2448],[3.4676, -0.6070, 1.6143],[4.0775, -0.7137, 0.8460],[4.5306, -0.7931, -0.0306],[4.8097, -0.8419, -0.9817],[0.9012, -0.3285, 3.0018],[1.7678, -0.6444, 2.7133],[2.5664, -0.9355, 2.2448],[3.2664, -1.1906, 1.6143],[3.8409, -1.4000, 0.8460],[4.2678, -1.5556, -0.0306],[4.5306, -1.6515, -0.9817],[0.8111, -0.4769, 3.0018],[1.5909, -0.9355, 2.7133],[2.3097, -1.3581, 2.2448],[2.9397, -1.7285, 1.6143],[3.4567, -2.0325, 0.8460],[3.8409, -2.2584, -0.0306],[4.0775, -2.3975, -0.9817],[0.6897, -0.6070, 3.0018],[1.3530, -1.1906, 2.7133],[1.9642, -1.7285, 2.2448],[2.5000, -2.2000, 1.6143],[2.9397, -2.5869, 0.8460],[3.2664, -2.8744, -0.0306],[3.4676, -3.0515, -0.9817],[0.5419, -0.7137, 3.0018],[1.0630, -1.4000, 2.7133],[1.5433, -2.0325, 2.2448],[1.9642, -2.5869, 1.6143],[2.3097, -3.0419, 0.8460],[2.5664, -3.3800, -0.0306],[2.7245, -3.5882, -0.9817],[0.3733, -0.7931, 3.0018],[0.7322, -1.5556, 2.7133],[1.0630, -2.2584, 2.2448],[1.3530, -2.8744, 1.6143],[1.5909, -3.3800, 0.8460],[1.7678, -3.7556, -0.0306],[1.8767, -3.9870, -0.9817],[0.1903, -0.8419, 3.0018],[0.3733, -1.6515, 2.7133],[0.5419, -2.3975, 2.2448],[0.6897, -3.0515, 1.6143],[0.8111, -3.5882, 0.8460],[0.9012, -3.9870, -0.0306],[0.9567, -4.2325, -0.9817],[-0.0000, -0.8584, 3.0018],[0.0000, -1.6838, 2.7133],[0.0000, -2.4445, 2.2448],[-0.0000, -3.1113, 1.6143],[-0.0000, -3.6585, 0.8460],[0.0000, -4.0651, -0.0306],[-0.0000, -4.3155, -0.9817],[-0.1903, -0.8419, 3.0018],[-0.3733, -1.6515, 2.7133],[-0.5419, -2.3975, 2.2448],[-0.6897, -3.0515, 1.6143],[-0.8111, -3.5882, 0.8460],[-0.9012, -3.9870, -0.0306],[-0.9567, -4.2325, -0.9817],[-0.3733, -0.7931, 3.0018],[-0.7322, -1.5556, 2.7133],[-1.0630, -2.2584, 2.2448],[-1.3530, -2.8744, 1.6143],[-1.5909, -3.3800, 0.8460],[-1.7678, -3.7556, -0.0306],[-1.8767, -3.9870, -0.9817],[-0.5419, -0.7137, 3.0018],[-1.0630, -1.4000, 2.7133],[-1.5433, -2.0325, 2.2448],[-1.9642, -2.5869, 1.6143],[-2.3097, -3.0419, 0.8460],[-2.5664, -3.3800, -0.0306],[-2.7245, -3.5882, -0.9817],[-0.0000, -0.0000, 3.0992],[-0.6897, -0.6070, 3.0018],[-1.3530, -1.1906, 2.7133],[-1.9642, -1.7285, 2.2448],[-2.5000, -2.2000, 1.6143],[-2.9397, -2.5869, 0.8460],[-3.2664, -2.8744, -0.0306],[-3.4676, -3.0515, -0.9817],[-0.8111, -0.4769, 3.0018],[-1.5909, -0.9355, 2.7133],[-2.3097, -1.3581, 2.2448],[-2.9397, -1.7285, 1.6143],[-3.4567, -2.0325, 0.8460],[-3.8409, -2.2584, -0.0306],[-4.0775, -2.3975, -0.9817],[-0.9012, -0.3285, 3.0018],[-1.7678, -0.6444, 2.7133],[-2.5664, -0.9355, 2.2448],[-3.2664, -1.1906, 1.6143],[-3.8409, -1.4000, 0.8460],[-4.2678, -1.5556, -0.0306],[-4.5306, -1.6515, -0.9817],[-0.9567, -0.1675, 3.0018],[-1.8767, -0.3285, 2.7133],[-2.7245, -0.4769, 2.2448],[-3.4676, -0.6070, 1.6143],[-4.0775, -0.7137, 0.8460],[-4.5306, -0.7931, -0.0306],[-4.8097, -0.8419, -0.9817],[-0.9755, -0.0000, 3.0018],[-1.9134, -0.0000, 2.7133],[-2.7779, -0.0000, 2.2448],[-3.5355, -0.0000, 1.6143],[-4.1573, -0.0000, 0.8460],[-4.6194, -0.0000, -0.0306],[-4.9039, -0.0000, -0.9817],[-0.9567, 0.1675, 3.0018],[-1.8767, 0.3285, 2.7133],[-2.7245, 0.4769, 2.2448],[-3.4676, 0.6070, 1.6143],[-4.0775, 0.7137, 0.8460],[-4.5306, 0.7931, -0.0306],[-4.8097, 0.8419, -0.9817],[-0.9012, 0.3285, 3.0018],[-1.7678, 0.6444, 2.7133],[-2.5664, 0.9355, 2.2448],[-3.2664, 1.1906, 1.6143],[-3.8409, 1.4000, 0.8460],[-4.2678, 1.5556, -0.0306],[-4.5306, 1.6515, -0.9817],[-0.8111, 0.4769, 3.0018],[-1.5909, 0.9355, 2.7133],[-2.3097, 1.3581, 2.2448],[-2.9397, 1.7285, 1.6143],[-3.4567, 2.0325, 0.8460],[-3.8409, 2.2584, -0.0306],[-4.0775, 2.3975, -0.9817],[-0.6897, 0.6070, 3.0018],[-1.3530, 1.1906, 2.7133],[-1.9642, 1.7285, 2.2448],[-2.5000, 2.2000, 1.6143],[-2.9397, 2.5869, 0.8460],[-3.2664, 2.8744, -0.0306],[-3.4676, 3.0515, -0.9817],[-0.5419, 0.7137, 3.0018],[-1.0630, 1.4000, 2.7133],[-1.5433, 2.0325, 2.2448],[-1.9642, 2.5869, 1.6143],[-2.3097, 3.0419, 0.8460],[-2.5664, 3.3800, -0.0306],[-2.7245, 3.5882, -0.9817],[-0.3733, 0.7931, 3.0018],[-0.7322, 1.5556, 2.7133],[-1.0630, 2.2584, 2.2448],[-1.3530, 2.8744, 1.6143],[-1.5909, 3.3800, 0.8460],[-1.7678, 3.7556, -0.0306],[-1.8766, 3.9870, -0.9817],[-0.1903, 0.8419, 3.0018],[-0.3733, 1.6515, 2.7133],[-0.5419, 2.3975, 2.2448],[-0.6897, 3.0515, 1.6143],[-0.8111, 3.5882, 0.8460],[-0.9012, 3.9870, -0.0306],[-0.9567, 4.2325, -0.9817]]

# CAM = [[0.,0.79999991,-1.3],[1.00000001,0.70000003,0],[0.,0.79999991,1.3],[-1.20000005,0,0],[0,1.39999998,0]]
CAM = [[0.14011652,0.46194088,1.03052575],[0.15396892,0.50175408,-1.2227992],[0.79485165,0.39933653,0.00140136],[0.68071215,0.18155913,0.02773174],[-0.03979695,0.92118744,0.18266273],[0.51290314,0.25953938,-0.6727925,],[-0.64931954,0.00425997,-0.40172596],[0.04464084,0.08686056,-1.35763202],[0.53587735,0.5726973,-0.29167617],[-0.77131011,0.22515772,1.10598779]]

K = np.array([[149.84375,0., -68.5,0.], [0., 149.84375, -68.5,0.], [  0.,0., -68.5,0.]])

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels
    K = Matrix(
        ((alpha_u, skew,    u_0),
        (   0  , alpha_v, v_0),
        (   0  , 0,     1 )))
    return K


def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))
    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location
    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam
    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))

def sample_hemisphere():
    # hemi_sphere = bpy.data.objects['Sphere']
    # indx = random.randint(0, len(hemi_sphere.data.vertices))
    # v = hemi_sphere.data.vertices[indx]
    # co_final = hemi_sphere.matrix_world @ v.co
    # return co_final
    indx = random.randint(0, len(HEMI_SPHERE))
    return HEMI_SPHERE[indx]

def get_cam(mat):
    location = -1 * np.matmul(np.transpose(mat[:,:3]), mat[:,3])
    T = np.zeros((3,1))
    T[2] = 1
    lookat = np.matmul(np.transpose(mat[:,:3]), T)
    return location, lookat

# ----------------------------------------------------------

PATH = "C:\\Users\\vikas\\Documents\\Brown_MSCS\\Research\\ShapeTextureRepresntations\\texture_fields\\test_data\\chair_1\\"

cam_dict = {
"world_mat_0": [],
"world_mat_1": [],
"world_mat_2": [],
"world_mat_3": [],
"world_mat_4": [],
"world_mat_5": [],
"world_mat_6": [],
"world_mat_7": [],
"world_mat_8": [],
"world_mat_9": [],
"camera_mat_0": [],
"camera_mat_1": [],
"camera_mat_2": [],
"camera_mat_3": [],
"camera_mat_4": [],
"camera_mat_5": [],
"camera_mat_6": [],
"camera_mat_7": [],
"camera_mat_8": [],
"camera_mat_9": []
}

# f = open(PATH+'depth\\cameras.npz')

cam = bpy.data.objects['Camera']

for num in range(0,10):
    # sample_loc = sample_hemisphere()
    sample_loc = CAM[num]
    num = str(num)
    cam.location = sample_loc
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    RT = np.asarray(RT)
    K = np.asarray(K)
    K_ = np.zeros((3,4))
    K_[:,0:3] = K
    cam_dict["camera_mat_"+num] = K_
    cam_dict["world_mat_"+num] = RT
    np.savez(PATH+'depth\\cameras.npz', **cam_dict)
    bpy.context.scene.node_tree.nodes["File Output"].base_path = PATH+'depth'
    bpy.context.scene.node_tree.nodes["File Output.001"].base_path = PATH+'image'
    bpy.ops.render.render()
    os.rename(PATH + 'depth' + '\\depth0001.exr ',  PATH + 'depth' + '\\00' + num + '.exr')
    os.rename(PATH + 'image' + '\\image0001.png ', PATH + 'image' + '\\00' + num + '.png')

for num in range(0,24):
    num = str(num)
    sample_loc = sample_hemisphere()
    cam.location = [i/3.5 for i in sample_loc]
    bpy.context.scene.node_tree.nodes["File Output.001"].base_path = PATH+'input_image'
    bpy.ops.render.render()
    os.rename(PATH + 'input_image' + '\\image0001.jpg ', PATH + 'input_image' + '\\00' + num + '.jpg')

pointcloud = {"points":[],"normals":[]}
C = bpy.context

points = []
for v in C.object.data.vertices:
    vert = v.co
    points.append(list(vert))

normals = []
for v in C.object.data.vertices:
    mx_inv = C.object.matrix_world.inverted()
    mx_norm = mx_inv.transposed().to_3x3()
    norm =  v.normal
    normals.append(list(norm))

pointcloud["points"] = points
pointcloud["normals"] = normals
np.savez(PATH+'cc067578ad92517bbe25370c898e25a5\\pointcloud.npz', **pointcloud)
