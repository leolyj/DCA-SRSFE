import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from utils_rigid import kabsch_transformation_estimation
import numpy as np
import copy
from pointnet2 import pointnet2_utils


EPS = 0.6  # used in DBSCAN for clustering
# 0.6 - waymo   0.4 - kitti   0.3 - lyft
L_K = 6  # num of neighboring points to calculate Laplacian Coordinates


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.point_criterion = nn.L1Loss(reduction='mean')

    def forward(self, x1, y1):
        '''
        n1 = n2 = 8192
        :param x1: (1, 3, 8192) -- (1, 8192, 3) -- (1, 1, 8192, 3)
        :param y1: (1, 3, 8192) -- (1, 8192, 3) -- (1, 8192, 1, 3)
        :return: loss
        '''

        x = torch.transpose(x1, 1, 2)
        y = torch.transpose(y1, 1, 2)

        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        dist = torch.sqrt(1e-8 + torch.sum(torch.pow(x - y, 2), 3))  # bs, ny, nx --- pointwise dist
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)
        
        return min1.mean() + min2.mean()

        
class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, x1, y1):
        dist = torch.sqrt(1e-6 + torch.sum(torch.pow(x1 - y1, 2), 1))
        return dist.mean()


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.point_criterion = nn.L1Loss(reduction='mean')

    def forward(self, x1, y1):
        return self.point_criterion(x1, y1)

      
def RigidReconstruction(pc, sf, min_samples, metric, eps, min_p_cluster):
    cluster_estimator = DBSCAN(min_samples=min_samples, metric=metric, eps=eps)
    labels_x = cluster_estimator.fit_predict(pc.transpose(1, 2).squeeze(0).cpu().numpy())
    
    clusters_x = []
    for class_label in np.unique(labels_x):
        if class_label != -1 and np.where(labels_x == class_label)[0].shape[0] >= min_p_cluster:
            clusters_x.append(np.where(labels_x == class_label)[0])
    
    pc_rigid = (pc + sf).transpose(1, 2).squeeze(0)  # (1, 8192, 3)
    for cluster_i in clusters_x:
        cluster_pc = pc[:, :, cluster_i].transpose(1, 2)  # pc
        cluster_sf = sf[:, :, cluster_i].transpose(1, 2)  # sf
        # cluster_recon = (cluster_pc + cluster_sf).squeeze(0)
        cluster_recon = (cluster_pc + cluster_sf)
        R_cluster, t_cluster, _, _ = kabsch_transformation_estimation(cluster_pc, cluster_recon)  # estimate R and T
        rigid_recon = (torch.matmul(R_cluster, cluster_pc.transpose(1, 2)) + t_cluster).detach().squeeze(0).transpose(0,1)
        pc_rigid[cluster_i, :] = rigid_recon
        
    return pc_rigid, clusters_x
    

class RigidLoss(nn.Module):
    def __init__(self, device, eps=EPS, min_samples=5, metric='euclidean', min_p_cluster=30):
        super(RigidLoss, self).__init__()
        self.device = device
        self.min_p_cluster = min_p_cluster
        self.cluster_estimator = DBSCAN(min_samples=min_samples, metric=metric, eps=eps)
        self.rigidity_criterion = nn.L1Loss(reduction='mean')

    #def forward(self, x, y, x0, y0, sf_x):
    def forward(self, pc ,sf, pc_t, sf_t):
        labels_x = self.cluster_estimator.fit_predict(pc.transpose(1, 2).squeeze(0).cpu().numpy())
        
        clusters_x = []
        for class_label in np.unique(labels_x):
            if class_label != -1 and np.where(labels_x == class_label)[0].shape[0] >= self.min_p_cluster:
                clusters_x.append(np.where(labels_x == class_label)[0])
                
        rigid_loss = torch.tensor(0.0).cuda().to(self.device)
        pc_rigid = (pc_t + sf_t).transpose(1, 2).squeeze(0)  # (1, 8196, 3)
        for cluster_i in clusters_x:
            cluster_pc = pc[:, :, cluster_i].transpose(1, 2)  # pc -- student input
            cluster_sf = sf[:, :, cluster_i].transpose(1, 2)  # sf -- student pred
            cluster_pc_t = pc_t[:, :, cluster_i].transpose(1, 2)  # pc_t -- teacher input
            cluster_sf_t = sf_t[:, :, cluster_i].transpose(1, 2)  # sf_t -- teacher pred
            cluster_recon = (cluster_pc + cluster_sf).squeeze(0)
            cluster_recon_t = (cluster_pc_t + cluster_sf_t).squeeze(0)
            
            R_cluster, t_cluster, _, _ = kabsch_transformation_estimation(cluster_pc_t, cluster_recon_t)  # estimate R and T
            rigid_recon_t = (torch.matmul(R_cluster, cluster_pc_t.transpose(1, 2)) + t_cluster).detach().squeeze(0).transpose(0,1)
            pc_rigid[cluster_i, :] = rigid_recon_t
            
            # rigid_loss += self.rigidity_criterion(cluster_recon, rigid_recon_t)
            
            # cluster_x_i, (cluster_x_i + sf_x_i)
            
        #pc_rigid = RigidReconstruction(pc_t, sf_t)
        rigid_loss = self.rigidity_criterion((pc + sf).transpose(1, 2).squeeze(0), pc_rigid)
            
        return rigid_loss

        
class ConsistLoss(nn.Module):
    def __init__(self):
        super(ConsistLoss, self).__init__()
        self.point_criterion = nn.L1Loss(reduction='mean')
        self.idx = 0

    def forward(self, input_t, sf_t, y1, pred):
        '''
        :param input_t: pc1_target             pc1_te
        :param sf_t:    teacher_predicted_sf   sf_te
        :param y1:      pc2_target             pc2_te
        :param pred:    student_prediction (pc1_target_2 + student_predicted_sf)  pred_st
        :return: loss
        '''

        # DR -- reconstructs rigid bodies
        rigid_recon, clusters = RigidReconstruction(input_t, sf_t, min_samples=5, metric='euclidean', eps=EPS, min_p_cluster=30)

        self.idx += 1
        # x1 = input_t + sf_t  # teacher_prediction (pc1_target + teacher_predicted_sf)
        x1 = torch.transpose(rigid_recon, 0, 1).unsqueeze(0)

        # x -- pc1+sf  (1, 3, 8192) -- (1, 8192, 3) -- (1, 1, 8192, 3)
        # y -- pc2     (1, 3, 8192) -- (1, 8192, 3) -- (1, 8192, 1, 3)
        x = torch.transpose(x1, 1, 2)
        y = torch.transpose(y1, 1, 2)

        x = x.unsqueeze(1)
        y = y.unsqueeze(2)

        # Correspondence Refinement
        squared_dist = torch.sum(torch.pow(x - y, 2), 3)  # bs, ny, nx
        _, k_idx = torch.topk(squared_dist, L_K, dim=1, largest=False, sorted=False)
        k_idx = k_idx.permute(0, 2, 1).contiguous()  # (1, 8192, 10)
        grouped_y = pointnet2_utils.grouping_operation(y1, k_idx.int()).permute(0, 2, 3, 1)
        laplace_y = torch.sum(grouped_y - x1.permute(0, 2, 1).unsqueeze(2), dim=2) / float(L_K - 1)

        squared_dist_self = torch.sum(torch.pow(x - x.permute(0, 2, 1, 3), 2), 3)
        _, k_idx_self = torch.topk(squared_dist_self, L_K, dim=1, largest=False, sorted=False)
        k_idx_self = k_idx_self.permute(0, 2, 1).contiguous()  # (1, 8192, 10)
        grouped_x = pointnet2_utils.grouping_operation(x1, k_idx_self.int()).permute(0, 2, 3, 1)
        laplace_x = torch.sum(grouped_x - x1.permute(0, 2, 1).unsqueeze(2), dim=2) / float(L_K - 1)
        
        laplace_reduced = laplace_x - laplace_y
        rigid_refine = copy.deepcopy(rigid_recon)
        for cluster_i in clusters:
            rigid_refine[cluster_i, :] = rigid_recon[cluster_i, :] - torch.mean(laplace_reduced[:, cluster_i, :],
                                                                                dim=1).squeeze(0)

        loss = self.point_criterion(rigid_refine.unsqueeze(0), pred.permute(0, 2, 1))

        return loss

        
def pred_refine(input_t, sf_t, y1):
    # teacher_input_pc1,  teacher_output,  input_pc2 
    x1 = input_t + sf_t
    
    x = torch.transpose(x1, 1, 2)
    y = torch.transpose(y1, 1, 2)  # x -- pc1+sf  y -- pc2

    x = x.unsqueeze(1)
    y = y.unsqueeze(2)

    squared_dist = torch.sum(torch.pow(x - y, 2), 3)  # bs, ny, nx
    _, k_idx = torch.topk(squared_dist, L_K, dim=1, largest=False, sorted=False)
    k_idx = k_idx.permute(0, 2, 1).contiguous()  # (1, 8192, 10)
    grouped_y = pointnet2_utils.grouping_operation(y1, k_idx.int()).permute(0, 2, 3, 1)
    laplace_y = torch.sum(grouped_y - x1.permute(0, 2, 1).unsqueeze(2), dim=2) / float(L_K - 1)
        
    squared_dist_self = torch.sum(torch.pow(x - x.permute(0, 2, 1, 3), 2), 3)
    _, k_idx_self = torch.topk(squared_dist_self, L_K, dim=1, largest=False, sorted=False)
    k_idx_self = k_idx_self.permute(0, 2, 1).contiguous()  # (1, 8192, 10)
    grouped_x = pointnet2_utils.grouping_operation(x1, k_idx_self.int()).permute(0, 2, 3, 1)
    laplace_x = torch.sum(grouped_x - x1.permute(0, 2, 1).unsqueeze(2), dim=2) / float(L_K - 1)
        
    laplace_reduced = laplace_x - laplace_y
    laplace_dist = torch.sum(torch.pow(laplace_reduced, 2), 2)
    x_zeros = torch.zeros_like(x1)
    laplace_dist = laplace_dist.unsqueeze(2).repeat(1, 1, 3)
    #x_confident = torch.where(laplace_dist < 0.2, x1.permute(0, 2, 1) - laplace_reduced, pred.permute(0, 2, 1))
    
    rigid_recon, clusters = RigidReconstruction(input_t, sf_t, min_samples=5, metric='euclidean', eps=EPS, min_p_cluster=30)
    for cluster_i in clusters:
        rigid_recon[cluster_i, :] = rigid_recon[cluster_i, :] - torch.mean(laplace_reduced[:, cluster_i, :], dim=1).squeeze(0)
        
    return rigid_recon


if __name__ == '__main__':
    pass
        
        
        
        
