print(__doc__)

import numpy as np

# from sklearn.cluster import DBSCAN
# import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import copy
import open3d as o3d



def get_data():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.xyz32f, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgra8, 30)

    profile = pipeline.start()

    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # print("DEPTH IMAGESSHAPOE ", depth_image.shape)

    # feature_image=np.reshape(color_image, (color_image.shape[0]*color_image.shape[1], color_image.shape[2]))

    pc = rs.pointcloud()
    points = rs.points
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    # color = frames.get_color_frame()
    # pc.map_to(color)
    points = pc.calculate(depth)
    vtx = np.asanyarray(points.get_vertices())
    print("Size of points: ", vtx.shape)
    tex = np.asanyarray(points.get_texture_coordinates())
    pipeline.stop()

    return points

def _save_point_cloud(point_cloud, counter):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    print("**************************DONR*************************")
    o3d.io.write_point_cloud("point_clouds/point_cloud_" + str(counter) + ".pcd", pcd)
    print("**************************DONR*************************")


def draw_registration_result(source, target, transformation, title=""):
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    source.transform(transformation)
    o3d.visualization.draw_geometries([source, target], window_name=title)


def draw_registration_result_axis(source, target, transformation, title):
    _draw_list = []
    source.paint_uniform_color([1, 0.706, 0])
    # target.paint_uniform_color([0, 0.651, 0.929])
    source.transform(transformation)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis_origin = copy.deepcopy(mesh).translate((0, 0, 0))
    axis_source = copy.deepcopy(mesh).translate(source.get_center())
    # axis_target = copy.deepcopy(mesh).translate(target.get_center())
    # o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])
    axis_origin.scale(0.1, center=(0, 0, 0))
    axis_source.scale(0.01, center=source.get_center())
    # axis_target.scale(0.01, center=target.get_center())
    _draw_list.append(axis_source)
    # _draw_list.append(axis_target)
    _draw_list.append(source)
    for _point in target.tolist():
        _tmp = copy.deepcopy(mesh).translate(_point)
        _tmp.scale(0.01, center=_point)
        _draw_list.append(_tmp)

def _draw_point_clouds(formatted_array, desired_point_array, transformation, title=""):
    if isinstance(formatted_array, np.ndarray):
        if desired_point_array.size != 0:
            _formatted_cloud = o3d.geometry.PointCloud()
            _desired_point_cloud = o3d.geometry.PointCloud()
            _formatted_cloud.points = o3d.utility.Vector3dVector(formatted_array)
            _desired_point_cloud.points = o3d.utility.Vector3dVector(desired_point_array)
            draw_registration_result_axis(_formatted_cloud, desired_point_array, transformation, title)
    elif isinstance(formatted_array, o3d.geometry.PointCloud):
        draw_registration_result(formatted_array, desired_point_array, transformation, title)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


if __name__=="__main__":
    points = get_data()
    _save_point_cloud(points,1)