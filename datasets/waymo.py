import sys, os
import os.path as osp
import numpy as np
import glob

import torch.utils.data as data

__all__ = ['waymo']


class waymo(data.Dataset):
    def __init__(self,
                 train,
                 transform,
                 gen_func,
                 args
                 ):
        self.root = args.data_root
        # assert train is False
        self.npoints = args.num_points
        if train:
            self.split = 'train'
        else:
            self.split = 'valid'
        self.train = train
        self.transform = transform
        self.gen_func = gen_func
        self.remove_ground = args.remove_ground
        self.num_points = args.num_points

        self.save_dir_path = os.path.join(self.root, 'waymo/scene_flow_data_npz')
        self.data_path = os.path.join(self.root, 'waymo_processed_data')
        self.label_path = os.path.join(self.root, 'scene_flow_label/' + self.split)
        self.split_dir = os.path.join(self.root, 'ImageSets/' + (self.split + '.txt'))
        self.sample_sequence_list = [x.strip() for x in open(self.split_dir).readlines()]

        # remove one segment which largely records ground points
        if 'segment-14383152291533557785_240_000_260_000_with_camera_labels.tfrecord' in self.sample_sequence_list:
            self.sample_sequence_list.remove('segment-14383152291533557785_240_000_260_000_with_camera_labels.tfrecord')
        
        print(self.split)
        print(len(self.sample_sequence_list))
        self.cache = {}
        self.samples = []

        for s in range(len(self.sample_sequence_list)):
            sequence_name_k = os.path.splitext(self.sample_sequence_list[s])[0]  # segment-xxx
            npz_dir = os.path.join(self.save_dir_path, sequence_name_k)
            npz_path = glob.glob(os.path.join(npz_dir, '*.npz'))
            self.samples += npz_path
        print(len(self.samples))
        self.samples = self.samples[:len(self.samples) // 2]
        print('chose ', len(self.samples))
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fn = self.samples[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1']
            pc2 = data['pc2']
            sf = data['flow'] * 0.1  # velocity to flow
            pose_1 = data['pose1']
            pose_2 = data['pose2']
            translation_1 = pose_1[0:3, 3]
            rotation_1 = pose_1[0:3, 0:3]
            rotation_inv_1 = np.linalg.inv(rotation_1)
            translation_2 = pose_2[0:3, 3]
            rotation_2 = pose_2[0:3, 0:3]
            rotation_inv_2 = np.linalg.inv(rotation_2)
            # add ego-motion for scene flow
            sf = pc2 - ((pc2 - sf) @ rotation_inv_2 + translation_2 - translation_1) @ rotation_1

        pc1_x90_idx = pc1[:, 0] > abs(pc1[:, 1])  # front_only
        pc2_x90_idx = pc2[:, 0] > abs(pc2[:, 1])
        not_ground_1 = pc1[:, 2] > 0.3  # remove ground points
        not_ground_2 = pc2[:, 2] > 0.3
        not_outrange_1 = np.square(pc1[:, 0]) + np.square(pc1[:, 1]) + np.square(pc1[:, 2]) < 60 * 60  # remove outrange
        not_outrange_2 = np.square(pc2[:, 0]) + np.square(pc2[:, 1]) + np.square(pc2[:, 2]) < 60 * 60
        not_outbound_1 = abs(pc1[:, 1]) < 50  # left and right bound
        not_outbound_2 = abs(pc2[:, 1]) < 50
        select_idx_1 = pc1_x90_idx * not_ground_1 * not_outrange_1 * not_outbound_1  # selected index
        select_idx_2 = pc2_x90_idx * not_ground_2 * not_outrange_2 * not_outbound_2
        pc1_transformed = pc2[select_idx_2, :]  # change order
        pc2_transformed = pc1[select_idx_1, :]
        sf_transformed = -sf[select_idx_2, :]  # reverse (pc2 -> pc1)

        n1 = pc1_transformed.shape[0]
        n2 = pc2_transformed.shape[0]
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)),
                                         axis=-1)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)),
                                         axis=-1)

        pc1_transformed = pc1_transformed[sample_idx1, :]
        pc2_transformed = pc2_transformed[sample_idx2, :]
        sf_transformed = sf_transformed[sample_idx1, :]  ###
        pc1_transformed = np.vstack((pc1_transformed[:, 1], pc1_transformed[:, 2], pc1_transformed[:, 0])).T
        pc2_transformed = np.vstack((pc2_transformed[:, 1], pc2_transformed[:, 2], pc2_transformed[:, 0])).T
        sf_transformed = np.vstack((sf_transformed[:, 1], sf_transformed[:, 2], sf_transformed[:, 0])).T
        # pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        # pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])

        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1, pc2, sf, generated_data = self.gen_func([pc1_transformed,
                                                      pc2_transformed,
                                                      sf_transformed])

        return pc1, pc2, sf, generated_data, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
