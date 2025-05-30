# from toolbox import EdgeProcessorTool as Tool
import cv2
import numpy as np
import math
import os
import json
from matplotlib import pyplot as plt

class GetImageEigenvector:
    """获取图像的Hu矩，距离描述子，角度描述子，归一化长宽比合并的特征描述向量，输入处理完毕的轮廓填充图像"""
    NUM_POINTS = 100  # 均匀采样的点数
    THETA_BINS = 36  # 角度描述子分箱数
    DISTANCE_BINS = 25  # 距离描述子分箱数
    def __init__(self,dict_obj:dict):
        self.dict_obj = dict_obj
        # self.image = self.dict_obj["mask"]
        self.contour  = self.dict_obj["contour"]
        self.bbox = self.dict_obj["bbox"]
        self.shape = self.dict_obj["shape"]
        self.theta = self.dict_obj["theta"]
        self.image = self.dict_obj["mask"]
        self.center = self.dict_obj["center"]

    def _get_hu_moments(self):
        """获取图像的Hu矩"""
        moments = cv2.moments(self.image)
        # cv2.imshow("image", self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        hu_moments = cv2.HuMoments(moments).flatten()
        # 对Hu矩进行对数变换
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
        #对第七个Hu矩进行取绝对值
        hu_moments[6] = np.abs(hu_moments[6])
        return hu_moments
    def _get_distance_descriptor(self):
        """获取图像的距离描述子"""
        num_points = min(len(self.contour), self.NUM_POINTS)
        #均匀采样轮廓点
        sampled_points = self._sample_contour_evenly(self.contour, num_points)
        #以中心点为原点，计算每个采样点到中心点的距离
        distances = [math.sqrt((point[0] - self.center[0]) ** 2 + (point[1] - self.center[1]) ** 2) for point in sampled_points]
        # 归一化距离
        max_distance = max(distances) if distances else 1.0
        normalized_distances = [d / max_distance for d in distances]
        # 将距离分成指定数量的区间
        distance_bins = np.linspace(0, 1, self.DISTANCE_BINS + 1)
        # 计算每个距离落在哪个区间
        distance_histogram, _ = np.histogram(normalized_distances, bins=distance_bins)
        distance_histogram = distance_histogram.astype(np.float32)
        normalized_histogram = distance_histogram/num_points

        return normalized_histogram

    def _get_angle_descriptor(self):
        """获取图像的角度描述子"""
        num_points = min(len(self.contour), self.NUM_POINTS)
        #均匀采样轮廓点
        sampled_points = self._sample_contour_evenly(self.contour, num_points)
        # 计算每个采样点相对于中心点的角度
        #从self.center出发，顺时针计算每个采样点的相对角度
        relative_angles = self._get_relative_angle(sampled_points)
        # 将角度分成指定数量的区间
        angle_bins = np.linspace(0, 360, self.THETA_BINS + 1)
        # 计算每个角度落在哪个区间
        angle_histogram, _ = np.histogram(relative_angles, bins=angle_bins)
        angle_histogram = angle_histogram.astype(np.float32)
        normalized_histogram = angle_histogram / num_points
        return normalized_histogram
    
    def _get_relative_angle(self, sampled_points):
        """从主角度出发，计算每个采样点的相对角度"""
        relative_angles = []
        
        for point in sampled_points:
            # 计算点相对于中心的绝对角度
            absolute_angle = math.atan2(point[1] - self.center[1], point[0] - self.center[0])
            absolute_angle = math.degrees(absolute_angle)  # 转换为度数
            
            # 计算相对于主方向的角度
            relative_angle = absolute_angle - self.theta
            
            # 将角度归一化到 [-180, 180] 范围
            # relative_angle = ((relative_angle + 180) % 360) - 180
            
            # 归一化到 [0, 360] 范围
            relative_angle = relative_angle % 360 
            
            relative_angles.append(relative_angle)
        
        return relative_angles


    def _sample_contour_evenly(self, contour, num_points):
        """
        均匀采样轮廓，返回指定数量的点
        :param contour: 输入轮廓 (np.array, shape=(N,1,2))
        :param num_points: 要采样的点数
        :return: 均匀采样后的点集 (list of tuples)
        """
        idx = np.round(np.linspace(0, len(contour) - 1, num_points)).astype(int)
        return contour[idx, 0, :]

    def _get_aspect_ratio(self):
        """获取图像的归一化长宽比"""
        width = self.shape[0]
        height = self.shape[1]
        if width == 0 or height == 0:
            return 0.0
        aspect_ratio = width / height
        return aspect_ratio
    
    def __call__(self):
        """获取图像的特征描述向量"""
        hu_moments = self._get_hu_moments()
        distance_descriptor = self._get_distance_descriptor()
        angle_descriptor = self._get_angle_descriptor()
        aspect_ratio = self._get_aspect_ratio()
        # Min-Max归一化到[0,1]范围 - 适合L1距离
        hu_min, hu_max = hu_moments.min(), hu_moments.max()
        hu_moments_norm = (hu_moments - hu_min) / (hu_max - hu_min + 1e-7)
        
        dist_min, dist_max = distance_descriptor.min(), distance_descriptor.max()
        distance_descriptor_norm = (distance_descriptor - dist_min) / (dist_max - dist_min + 1e-7)
        
        angle_min, angle_max = angle_descriptor.min(), angle_descriptor.max()
        angle_descriptor_norm = (angle_descriptor - angle_min) / (angle_max - angle_min + 1e-7)
        # 合并所有特征描述子
        # feature_vector = np.concatenate((hu_moments, distance_descriptor, angle_descriptor, [aspect_ratio])).astype(np.float32)
        # 重塑为 (1, 69) 形状并确保数据类型为 float
        # feature_vector = feature_vector.reshape(1, -1).astype(np.float32)
        feature_vector = np.concatenate((hu_moments_norm, distance_descriptor_norm, 
                                   angle_descriptor_norm, [aspect_ratio])).astype(np.float32)
        return feature_vector