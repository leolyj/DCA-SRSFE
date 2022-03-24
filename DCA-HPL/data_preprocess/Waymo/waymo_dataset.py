'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
# import json
import numpy as np
import sys
import pickle
import glob
from helper_ply import write_ply, read_ply


class SceneflowDataset():
    def __init__(self, root_path='/mnt/3t/ST3D/ST3D-master/data/waymo', npoints=16384, train=False):
        self.root_path = root_path
        self.npoints = npoints
        if train:
            self.split = 'train'
        else:
            self.split = 'valid'
        self.save_dir_path = os.path.join(self.root_path, 'scene_flow_data_npz')
        self.data_path = os.path.join(self.root_path, 'waymo_processed_data')
        self.label_path = os.path.join(self.root_path, 'scene_flow_label/' + self.split)
        self.split_dir = os.path.join(self.root_path, 'ImageSets/' + (self.split + '.txt'))
        self.sample_sequence_list = [x.strip() for x in open(self.split_dir).readlines()]
        # self.sample_sequence_list = ['segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord']
        print(self.split)
        print(len(self.sample_sequence_list))
        self.cache = {}
        self.datapath = []

        for s in range(len(self.sample_sequence_list)):
            sequence_name_k = os.path.splitext(self.sample_sequence_list[s])[0]  # segment-xxx
            npz_dir = os.path.join(self.save_dir_path, sequence_name_k)
            npz_path = glob.glob(os.path.join(npz_dir, '*.npz'))
            self.datapath += npz_path
        print(len(self.datapath))

    def convert_waymo_data(self):
        print('Loading Waymo dataset')
        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name_k = os.path.splitext(self.sample_sequence_list[k])[0]  # segment-xxx
            print(sequence_name_k)
            seq_path_k = os.path.join(self.data_path, sequence_name_k)  # waymo_processed_data/segment../
            label_path_k = os.path.join(self.label_path, sequence_name_k)  # scene_flow_label/train/segment../
            savez_path_k = os.path.join(self.save_dir_path, sequence_name_k)  # scene_flow_data_npz/segment../
            if not os.path.exists(savez_path_k):
                os.mkdir(savez_path_k)
            # else:
            #     continue

            if not os.path.exists(seq_path_k):
                num_skipped_infos += 1
                print('not exist: ', seq_path_k)
                print('*********************************\n')
                continue

            npy_file_list = glob.glob(os.path.join(seq_path_k, '*.npy'))
            info_file = glob.glob(os.path.join(seq_path_k, '*.pkl'))[0]  # xxx.pkl
            with open(info_file, 'rb') as f:
                labels = pickle.load(f)

            npy_file_list.sort(key=lambda x: int(x.split('/')[-1][:-4]))
            # print(npy_file_list)
            for idx in range(len(npy_file_list) - 1):
                info_1 = labels[idx] # ['point_cloud', 'frame_id', 'image', 'pose', 'annos', 'num_points_of_each_lidar']
                pose_1 = info_1['pose']  # 4x4
                name_from_info1 = '%04d' % info_1['point_cloud']['sample_idx']
                info_2 = labels[idx + 1]
                pose_2 = info_2['pose']  # 4x4
                name_from_info2 = '%04d' % info_2['point_cloud']['sample_idx']

                pc1_path = npy_file_list[idx]
                pc2_path = npy_file_list[idx + 1]
                pc1_name = pc1_path.split('/')[-1][:-4]
                pc2_name = pc2_path.split('/')[-1][:-4]
                flow_path = os.path.join(label_path_k, pc2_name + '.ply')

                if name_from_info1 != pc1_name:
                    print('error!')
                    print('info1: ', name_from_info1, 'pc1: ', pc1_name)
                    exit(1)
                if name_from_info2 != pc2_name:
                    print('error!')
                    print('info2: ', name_from_info2, 'pc2: ', pc2_name)
                    exit(1)

                pc1_data, flag1 = self.get_data(pc1_path)
                pc2_data, flag2 = self.get_data(pc2_path)
                flow_data = self.get_label(flow_path, flag2)
                savez_path = os.path.join(savez_path_k, pc1_name + '_' + pc2_name + '.npz')
                print(savez_path)
                np.savez_compressed(savez_path, pc1=pc1_data, pc2=pc2_data, flow=flow_data, pose1=pose_1, pose2=pose_2)
                
    def get_data(self, lidar_file):
        point_features = np.load(lidar_file)  # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]

        points, intensity, elongation, NLZ_flag = point_features[:, 0:3], point_features[:, 3], point_features[:, 4], point_features[:, 5]
        points = points[NLZ_flag == -1]
        # intensity = np.tanh(intensity)
        return points, NLZ_flag

    def get_label(self, label_file, flags):
        flow_data = read_ply(label_file)
        flow = np.vstack((flow_data['x'], flow_data['y'], flow_data['z'])).T
        return flow[flags == -1]

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pc1 = data['pc1']
                pc2 = data['pc2']
                flow = data['flow']
                # pc1 = data['pc2']
                # pc2 = data['pc1']
                # flow = data['flow']
            n1 = pc1.shape[0]
            n2 = pc2.shape[0]
            if n1 >= self.npoints:
                sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
            if n2 >= self.npoints:
                sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

            pc1_ = np.copy(pc1)[sample_idx1, :]
            pc2_ = np.copy(pc2)[sample_idx2, :]
            flow_ = np.copy(flow)[sample_idx2, :]
            # flow_ = np.copy(flow)[sample_idx1, :]

        color1 = np.zeros([self.npoints, 3])
        color2 = np.zeros([self.npoints, 3])
        mask = np.ones([self.npoints])

        return pc1_, pc2_, color1, color2, flow_, mask

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    # sf_dataset = SceneflowDataset(root_path='/mnt/3t/ST3D-master/data/waymo', split='train')
    # sf_dataset.convert_waymo_data()
    # sf_dataset = SceneflowDataset(root_path='/mnt/3t/ST3D-master/data/waymo', train=True)
    sf_dataset = SceneflowDataset(root_path='/mnt/3t/ST3D/ST3D-master/data/waymo', train=False)  # valid

    sf_dataset.convert_waymo_data()



