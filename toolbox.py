import cv2
import numpy as np
import os
import json
import random
import math
import matplotlib.pyplot as plt


#边缘处理
class EdgeProcessorTool:
    """一个用来处理边缘的工具类，可选是否亮于背景，返回全部轮廓还是最外层轮廓，输入为"""
    def __init__(self,polar = "light", outer = True):
        polar_dict = {
            "light": cv2.THRESH_BINARY,  # 亮于背景
            "dark": cv2.THRESH_BINARY_INV,   # 暗于背景
        }
        outer_dict = {
            True: cv2.RETR_EXTERNAL,  # 只返回最外层轮廓
            False: cv2.RETR_LIST,     # 返回所有轮廓
        }
        self.polar = polar_dict[polar]
        self.outer = outer_dict[outer] 

    def _process_to_b(self, image):
        """处理图像到二值图"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #区域自适应阈值处理
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, self.polar, 11, 2)
        return binary,gray
    
    def _get_and_process_contours(self, canny_img):
        """获取轮廓并处理,返回轮廓，质心,矩方向,最小外接矩形,bbox,填充完毕的mask,理论单目标，只处理最外层轮廓"""
        contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #获取图像宽高
        height, width = canny_img.shape[:2] 
        mask = np.zeros(canny_img.shape, dtype=np.uint8)
        contours_filtered = []
        for contour in contours:
            if len(contour) < 5:
                continue
            #计算轮廓的最小矩形框，与坐标平行的bbox
            x, y, w, h = cv2.boundingRect(contour)
            #如果这个框的长宽与原图相似，则认为框住了整个图像
            if w / width > 0.9 and h / height > 0.9: 
                continue
            contours_filtered.append(contour)
        #筛选最大的轮廓
        if len(contours_filtered) == 0:
            return [], None, None, None, None, None
        contours_filtered = sorted(contours_filtered, key=cv2.contourArea, reverse=True)
        contour = contours_filtered[0]#抽取最大轮廓
        #获取bbox
        x, y, w, h = cv2.boundingRect(contour)
        #计算轮廓的最小外接矩形
        rect = cv2.minAreaRect(contour)
        shape = rect[1]
        #让宽最短，长最长
        if shape[0] > shape[1]:
            shape_wh = (shape[1], shape[0])
        else:
            shape_wh = shape
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        bbox = (x, y, w, h)
        #计算轮廓的质心
        M = cv2.moments(mask)
        if M["m00"] == 0:
            cx, cy = 0, 0
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            mu20 = M["m20"] / M["m00"] 
            mu02 = M["m02"] / M["m00"] 
            mu11 = M["m11"] / M["m00"] 
        center = (cx, cy)
        #计算轮廓的方向
        if abs(mu20 - mu02) > 1e-6:  # 避免除以零
            theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        else:
            theta = 0

        return contour, center, theta, shape_wh, bbox, mask
    def get_contours_image(self, image):
        binary, gray = self._process_to_b(image)
        """返回筛选过的bbox，最小轮廓的外接矩形，膨胀腐蚀处理过的canny全幅图像，根据canny抽取的所有轮廓点"""
        contours, _ = cv2.findContours(binary, self.outer, cv2.CHAIN_APPROX_SIMPLE)
        height, width = binary.shape[:2]
        mask_img = np.zeros(binary.shape, dtype=np.uint8)
        contours_list = []
        center_list = []
        theta_list = []
        shape_list = []
        bbox_list = []
        for contour in contours:
            if len(contour) < 5:
                continue
            #计算轮廓的最小矩形框，与坐标平行的bbox
            x, y, w, h = cv2.boundingRect(contour)
            #如果这个框的长宽与原图相似，则认为框住了整个图像
            if w / width > 0.9 and h / height > 0.9: 
                continue
            #框出原图
            mask = np.zeros(gray.shape, np.uint8)
            cut_image = gray[y:y+h, x:x+w]
            #进行otsu二值化处理，获取分割阈值
            retval,_ = cv2.threshold(cut_image, 0, 255, self.polar + cv2.THRESH_OTSU)
            max_thresh = min(int(retval * 1.5), 255)
            min_thresh = max(int(retval * 0.5), 0)
            canny_img = cv2.Canny(cut_image, min_thresh, max_thresh)
            #膨胀腐蚀处理
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            canny_img = cv2.dilate(canny_img, kernel_dilate, iterations=1)
            canny_img = cv2.erode(canny_img, kernel_erode, iterations=1)
            #贴回原图
            mask[y:y+h, x:x+w] = canny_img
            contour, center, theta, shape_wh, bbox, mask = self._get_and_process_contours(mask)
            if contour is None:
                continue
            contours_list.append(contour)
            center_list.append(center)
            theta_list.append(theta)
            shape_list.append(shape_wh)
            bbox_list.append(bbox)
            #将mask添加到mask_img中,单通道
            cv2.drawContours(mask_img, [contour], -1, 255, thickness=cv2.FILLED)
        
        return contours_list, center_list, theta_list, shape_list, bbox_list, mask_img