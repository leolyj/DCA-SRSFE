from lyft_dataset_sdk.lyftdataset import LyftDataset
from pcdet.datasets.lyft import lyft_utils
import os
import os.path
import numpy as np
from pyquaternion import Quaternion
from pcdet.datasets.nuscenes.helper_ply import write_ply, read_ply
import copy
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import torch


def remove_ego_points(points, center_radius=1.0):
    mask = ~((np.abs(points[:, 0]) < center_radius * 1.5) & (np.abs(points[:, 1]) < center_radius))
    return points[mask]


def get_points(lidar_path):
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)
    if points.shape[0] % 5 != 0:
        points = points[: points.shape[0] - (points.shape[0] % 5)]
    points = points.reshape([-1, 5])[:, :4]
    points = remove_ego_points(points, center_radius=1.5)
    return points[:, :3]


save_dir = '/media/asus/ef19a603-a137-4f24-91d3-0071b4004fdb/lyft_scene_flow'

root_path = '/media/asus/ef19a603-a137-4f24-91d3-0071b4004fdb/ST3D/ST3D-master/data/lyft'
version = 'trainval'  # ['trainval', 'one_scene', 'test']
data_path = os.path.join(root_path, version)
split_path = os.path.join(root_path, 'ImageSets')

assert version in ['trainval', 'one_scene', 'test']

if version == 'trainval':
    train_split_path = os.path.join(split_path, 'train.txt')
    val_split_path = os.path.join(split_path, 'val.txt')
elif version == 'test':
    train_split_path = ''
    val_split_path = os.path.join(split_path, 'test.txt')
elif version == 'one_scene':
    train_split_path = os.path.join(split_path, 'one_scene.txt')
    val_split_path = os.path.join(split_path, 'one_scene.txt')
else:
    raise NotImplementedError

train_scenes = [x.strip() for x in open(train_split_path).readlines()] if os.path.exists(train_split_path) else []
val_scenes = [x.strip() for x in open(val_split_path).readlines()] if os.path.exists(val_split_path) else []

lyft = LyftDataset(json_path=data_path + '/data', data_path=data_path, verbose=True)

available_scenes = lyft_utils.get_available_scenes(lyft)
available_scene_names = [s['name'] for s in available_scenes]
train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

ref_chans = ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]
for s_name in val_scenes:
    print(s_name)
    idx = available_scene_names.index(s_name)
    scene = available_scenes[idx]
    current_sample_token = scene['first_sample_token']
    num_frame = 1

    while current_sample_token != '':
        # print(num_frame)
        sample_record = lyft.get('sample', current_sample_token)
        # print(sample_record)
        sd_rec = lyft.get('sample_data', sample_record['data']['LIDAR_TOP'])
        '''
        for ref_chan in ref_chans:
            sample_data_token = sample_record['data'][ref_chan]
            print(sample_data_token)
            if ref_chan == 'LIDAR_TOP':
                lyft.render_sample_data(sample_data_token=sample_data_token, out_path='/media/asus/ef19a603-a137-4f24-91d3-0071b4004fdb/ST3D/ST3D-master/data/lyft')
                lidar_path = lyft.get_sample_data_path(sample_data_token)
                print(lidar_path)
        '''
        lidar_path = lyft.get_sample_data_path(sd_rec['token'])
        ref_boxes_1 = lyft.get_boxes(sd_rec['token'])
        # print(lidar_path)
        # print('ref_boxes_1', ref_boxes_1)

        cs_rec = lyft.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_rec = lyft.get("ego_pose", sd_rec["ego_pose_token"])
        # print(Quaternion(cs_rec['rotation']).rotation_matrix, cs_rec['translation'])
        # print(Quaternion(pose_rec['rotation']).rotation_matrix, pose_rec['translation'])

        points = get_points(lidar_path)
        # print(points.shape)
        # ply_file_1 = '/home/asus/jz/point_cloud_data/lyft/' + str(num_frame) + '_0.ply'
        # write_ply(ply_file_1, points[:, :3], ['x', 'y', 'z'])


        next_sample_token = sample_record['next']
        # print(next_sample_token)
        if next_sample_token == '':
            break
        next_sample_record = lyft.get('sample', next_sample_token)
        next_sd_rec = lyft.get('sample_data', next_sample_record['data']['LIDAR_TOP'])
        # next_lidar_path, next_boxes, _ = lyft.get_sample_data(next_sd_rec['token'])
        next_lidar_path = lyft.get_sample_data_path(next_sd_rec['token'])
        ref_boxes_2 = lyft.get_boxes(next_sd_rec['token'])
        next_cs_rec = lyft.get("calibrated_sensor", next_sd_rec["calibrated_sensor_token"])
        next_pose_rec = lyft.get("ego_pose", next_sd_rec["ego_pose_token"])
        next_points = get_points(next_lidar_path)

        # -------------------- ego-motion ------------------------------
        rot_sensor_1 = Quaternion(cs_rec['rotation']).rotation_matrix
        trans_sensor_1 = np.array(cs_rec['translation'])
        rot_sensor_inv_1 = np.linalg.inv(rot_sensor_1)
        rot_car_1 = Quaternion(pose_rec['rotation']).rotation_matrix
        trans_car_1 = np.array(pose_rec['translation'])
        rot_car_inv_1 = np.linalg.inv(rot_car_1)  # inverse rotation matrix

        rot_sensor_2 = Quaternion(next_cs_rec['rotation']).rotation_matrix
        trans_sensor_2 = np.array(next_cs_rec['translation'])
        rot_sensor_inv_2 = np.linalg.inv(rot_sensor_2)
        rot_car_2 = Quaternion(next_pose_rec['rotation']).rotation_matrix
        trans_car_2 = np.array(next_pose_rec['translation'])
        rot_car_inv_2 = np.linalg.inv(rot_car_2)  # inverse rotation matrix

        # -------------------- box-motion -------------------------------
        locs_1 = np.array([b.center for b in ref_boxes_1]).reshape(-1, 3)
        dims_1 = np.array([b.wlh for b in ref_boxes_1]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
        velocity_1 = np.array([b.velocity for b in ref_boxes_1]).reshape(-1, 3)
        rots_1 = np.array([lyft_utils.quaternion_yaw(b.orientation) for b in ref_boxes_1]).reshape(-1, 1)
        names_1 = np.array([b.name for b in ref_boxes_1])
        tokens_1 = np.array([b.token for b in ref_boxes_1])
        inst_tokens_1 = np.array([b.instance_token for b in ref_boxes_1])
        gt_boxes_1 = np.concatenate([locs_1, dims_1, rots_1, velocity_1[:, :2]], axis=1)

        locs_2 = np.array([b.center for b in ref_boxes_2]).reshape(-1, 3)
        dims_2 = np.array([b.wlh for b in ref_boxes_2]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
        velocity_2 = np.array([b.velocity for b in ref_boxes_2]).reshape(-1, 3)
        rots_2 = np.array([lyft_utils.quaternion_yaw(b.orientation) for b in ref_boxes_2]).reshape(-1, 1)
        names_2 = np.array([b.name for b in ref_boxes_2])
        tokens_2 = np.array([b.token for b in ref_boxes_2])
        inst_tokens_2 = np.array([b.instance_token for b in ref_boxes_2])
        gt_boxes_2 = np.concatenate([locs_2, dims_2, rots_2, velocity_2[:, :2]], axis=1)

        points_transform1 = copy.deepcopy(points)
        points_transform2 = copy.deepcopy(next_points)

        # ---------- sensor_coord -> car_coord -> global_coord ----------
        points_transform1 = (points_transform1 @ rot_sensor_inv_1) + trans_sensor_1
        points_transform2 = (points_transform2 @ rot_sensor_inv_2) + trans_sensor_2
        points_transform1 = (points_transform1 @ rot_car_inv_1) + trans_car_1
        points_transform2 = (points_transform2 @ rot_car_inv_2) + trans_car_2
        # ---------- box motion in global_coord ----------
        box_instance_1 = []
        box_center_1 = []
        box_wlh_1 = []
        box_orientation_1 = []
        for box_1 in ref_boxes_1:
            box_instance_1.append(box_1.instance_token)
            box_center_1.append(box_1.center)
            box_wlh_1.append(box_1.wlh)
            box_orientation_1.append(box_1.orientation)


        box_pts_idxs = roiaware_pool3d_utils.points_in_boxes_gpu(
            torch.from_numpy(points_transform2[:, 0:3]).unsqueeze(dim=0).float().cuda(),
            torch.from_numpy(gt_boxes_2[:, 0:7]).unsqueeze(dim=0).float().cuda()
        ).long().squeeze(dim=0).cpu().numpy()

        rgb_2 = 255 * np.ones_like(points_transform2, dtype=np.int32)
        for n, box_2 in enumerate(ref_boxes_2):
            instance_n = box_2.instance_token
            if instance_n in box_instance_1:
                idx = box_instance_1.index(instance_n)
                loc_1 = box_center_1[idx]
                loc_2 = box_2.center
                trans_1_2 = loc_1 - loc_2
                # ang_1 = box_orientation_1[idx].radians
                # ang_2 = box_2.orientation.radians
                ang_1 = lyft_utils.quaternion_yaw(box_orientation_1[idx])
                ang_2 = lyft_utils.quaternion_yaw(box_2.orientation)
                angle = (ang_2 - ang_1)
                rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                                       [np.sin(angle), np.cos(angle), 0],
                                       [0, 0, 1]])
                # if angle != 0:
                # print(n, trans_1_2, angle)
                points_transform2[box_pts_idxs == n] = (points_transform2[
                                                                box_pts_idxs == n] - loc_2) @ rotation_z + loc_1
                rgb_2[box_pts_idxs == n] = [255, 0, 0]

                # if (num_frame == 35) and (instance_n == 'c8ac146d6cef45bfadbec2f08737c79f'):
                #     print('num pts', sum(box_pts_idxs == n))
        # print('total', sum(rgb_2 == [255, 0, 0]))


        points_transform1 = (((points_transform1 - trans_car_1) @ rot_car_1) - trans_sensor_1) @ rot_sensor_1
        points_transform2 = (((points_transform2 - trans_car_1) @ rot_car_1) - trans_sensor_1) @ rot_sensor_1

        sf = (points_transform2 - next_points).astype(np.float32)  # label
        ply_file_1 = '/home/asus/jz/point_cloud_data/lyft/lyft_test/' + str(num_frame) + '_1.ply'
        write_ply(ply_file_1, points[:, :3], ['x', 'y', 'z'])

        ply_file_2 = '/home/asus/jz/point_cloud_data/lyft/lyft_test/' + str(num_frame) + '_2.ply'
        write_ply(ply_file_2, next_points[:, :3], ['x', 'y', 'z'])

        ply_file_t = '/home/asus/jz/point_cloud_data/lyft/lyft_test/' + str(num_frame) + '_t.ply'
        write_ply(ply_file_t, [next_points[:, :3] + sf, rgb_2], ['x', 'y', 'z', 'red', 'green', 'blue'])

        not_ground_1 = points[:, -1] > (0.3 - trans_sensor_1[2])
        not_ground_2 = next_points[:, -1] > (0.3 - trans_sensor_2[2])
        is_ground_1 = points[:, -1] < (0.3 - trans_sensor_1[2])
        is_ground_2 = next_points[:, -1] < (0.3 - trans_sensor_2[2])
        front_1 = points[:, 0] < - abs(points[:, 1])
        front_2 = next_points[:, 0] < - abs(next_points[:, 1])
        ply_file_f = '/home/asus/jz/point_cloud_data/lyft/lyft_test/' + str(num_frame) + '_front.ply'
        write_ply(ply_file_f, points[:, :3][front_1], ['x', 'y', 'z'])
        remove_1 = not_ground_1 * front_1
        remove_2 = not_ground_2 * front_2
        ply_file_r = '/home/asus/jz/point_cloud_data/lyft/lyft_test/' + str(num_frame) + '_remove.ply'
        write_ply(ply_file_r, points[:, :3][remove_1], ['x', 'y', 'z'])
        ply_file_r = '/home/asus/jz/point_cloud_data/lyft/lyft_test/' + str(num_frame) + '_removet.ply'
        write_ply(ply_file_r, next_points[:, :3][remove_2] + sf[remove_2], ['x', 'y', 'z'])

        save_path_dir = os.path.join(os.path.join(save_dir, version), s_name)
        if not os.path.exists(save_path_dir):
            os.mkdir(save_path_dir)
        savez_path = os.path.join(save_path_dir, str(num_frame) + '.npz')
        print(savez_path)
        # np.savez_compressed(savez_path, pc1=points[:, :3], pc2=next_points[:, :3], flow=sf, is_ground_1=is_ground_1,
        #                     is_ground_2=is_ground_2)  # is_ground
        # np.savez_compressed(savez_path, pc1=points[:, :3][remove_1], pc2=next_points[:, :3][remove_2],
        #                     flow=sf[remove_2])  # is_ground


        current_sample_token = next_sample_token

        num_frame += 1














