import os
import glob
# import pickle
# import copy
import numpy as np
# import torch
# import multiprocessing
# from tqdm import tqdm
# from pathlib import Path
# from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
# from pcdet.utils import box_utils, common_utils
# from pcdet.datasets.dataset import DatasetTemplate
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from os.path import join
from helper_ply import write_ply, read_ply
import zipfile

import argparse


parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--train_dir', type=str, default='/data/scene_flow/train', help='path to the waymo train split')
parser.add_argument('--valid_dir', type=str, default='/data/scene_flow/valid', help='path to the waymo valid split')
args = parser.parse_args()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_dir = args.train_dir
valid_dir = args.valid_dir

train_seq_list = []
valid_seq_list = []

for train_seq_path in glob.glob(join(train_dir, '*.tfrecord')):
    train_seq_list.append(train_seq_path)

for valid_seq_path in glob.glob(join(valid_dir, '*.tfrecord')):
    valid_seq_list.append(valid_seq_path)


def zip_folder(dirpath, outFullName):
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        fpath = path.replace(dirpath, '')

        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def parse_range_image_flow_and_camera_projection(frame):
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_flow_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_flow_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy()))

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
        if len(laser.ri_return2.range_image_flow_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_flow_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, range_image_top_pose


def convert_range_image_to_point_cloud_flow(frame, range_images, range_images_flow, camera_projections, range_image_top_pose, ri_index=0):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Convert range images flow to scene flow.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        range_imaages_flow: A dict similar to range_images.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        points_flow: {[N, 3]} list of scene flow vector of each point.
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """    
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []
    points_flow = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_flow = range_images_flow[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_flow_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_flow.data), range_image_flow.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_NLZ = range_image_tensor[..., 3]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]

        flow_x = range_image_flow_tensor[..., 0]
        flow_y = range_image_flow_tensor[..., 1]
        flow_z = range_image_flow_tensor[..., 2]
        flow_class = range_image_flow_tensor[..., 3]

        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.where(range_image_mask))
        points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
        points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
        points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))

        points_flow_x_tensor = tf.expand_dims(tf.gather_nd(flow_x, tf.compat.v1.where(range_image_mask)), axis=1)
        points_flow_y_tensor = tf.expand_dims(tf.gather_nd(flow_y, tf.compat.v1.where(range_image_mask)), axis=1)
        points_flow_z_tensor = tf.expand_dims(tf.gather_nd(flow_z, tf.compat.v1.where(range_image_mask)), axis=1)
        points_flow_class_tensor = tf.expand_dims(tf.gather_nd(flow_class, tf.compat.v1.where(range_image_mask)), axis=1)

        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        points_NLZ.append(points_NLZ_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())

        # points_flow.append(tf.concat([points_flow_x_tensor, points_flow_y_tensor, points_flow_z_tensor,
        #                              points_flow_class_tensor], axis=-1).numpy())
        points_flow.append(tf.concat([points_flow_x_tensor, points_flow_y_tensor, points_flow_z_tensor], axis=-1).numpy())

    return points, points_flow, cp_points, points_NLZ, points_intensity, points_elongation


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_NLZ = range_image_tensor[..., 3]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.where(range_image_mask))
        points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
        points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
        points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        points_NLZ.append(points_NLZ_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def gen_func(split='train'):
    if split == 'train':
        dir_path = join(train_dir, 'flow')
        seq_list = train_seq_list
    elif split == 'valid':
        dir_path = join(valid_dir, 'flow')
        seq_list = valid_seq_list
    print(dir_path)
    for sequence_file in seq_list:
        dir_path_i = join(dir_path, sequence_file.split('/')[-1][:-9])
        if not os.path.exists(dir_path_i):
            os.mkdir(dir_path_i)
            print(dir_path_i)  # '/data/scene_flow/train/flow/segment-xxx/'
        else:
            print('pass')
            continue
        dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
        print(dataset)
        flows_all_list = []
        for cnt, data in enumerate(dataset):
            if cnt < 0:
                continue
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            range_images, camera_projections, range_image_top_pose = \
                frame_utils.parse_range_image_and_camera_projection(frame)

            range_images_flow, _, _ = parse_range_image_flow_and_camera_projection(frame)
            points, flows, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud_flow(frame, range_images, range_images_flow, camera_projections, range_image_top_pose)

            points_all = np.concatenate(points, axis=0)  # points
            print(points_all.shape)
            flows_all = np.concatenate(flows, axis=0)  # scene flow
            print(flows_all.shape)

            points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
            points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
            points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

            num_points_of_each_lidar = [point.shape[0] for point in points]
            # save_points = np.concatenate([
            #     points_all, points_intensity, points_elongation, points_in_NLZ_flag
            # ], axis=-1).astype(np.float32)
            # save_points = np.concatenate([
            #     points_all, points_in_NLZ_flag
            # ], axis=-1).astype(np.float32)

            # np.save(cur_save_path, save_points)

            points = points_all[points_in_NLZ_flag == -1]

            ply_path = join(dir_path_i, 'ply')
            if not os.path.exists(ply_path):
                os.mkdir(ply_path)
            ply_file = join(ply_path, '%04d.ply' % cnt)
            print(ply_file)
            write_ply(ply_file, flows_all, ['x', 'y', 'z'])

        # zip_file = join(dir_path_i, dir_path_i.split('/')[-1] + '.zip')
        # zip_folder(join(dir_path_i, 'ply'), zip_file)


if __name__ == '__main__':
    print(tf.__version__)
    gen_func('train')
    gen_func('valid')

        

