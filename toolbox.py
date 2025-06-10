import cv2
import numpy as np
import os
import json
import random
import math
import matplotlib.pyplot as plt


#边缘处理
class EdgeProcessorTool:
    """一个用来处理边缘的工具类，可选是否亮于背景，返回全部轮廓还是最外层轮廓，输入图像,初始化时可选polar和outer参数"""
    # 类常量
    MIN_CONTOUR_POINTS = 5
    FRAME_RATIO_THRESHOLD = 0.9
    OTSU_MULTIPLIER_HIGH = 1.4
    OTSU_MULTIPLIER_LOW = 0.5
    DILATE_KERNEL_SIZE = (5, 5)
    ERODE_KERNEL_SIZE = (3, 3)
    def __init__(self,polar = "light", outer = True,skip_adaptive_threshold=False):
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
        self.otsu_multiplier_high = self.OTSU_MULTIPLIER_HIGH 
        self.otsu_multiplier_low = self.OTSU_MULTIPLIER_LOW 
        self.min_contour_points = self.MIN_CONTOUR_POINTS
        self.dilate_kernel_size = self.DILATE_KERNEL_SIZE
        self.erode_kernel_size = self.ERODE_KERNEL_SIZE
        self.frame_ratio_threshold = self.FRAME_RATIO_THRESHOLD
        #初始化膨胀和腐蚀的kernel
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.dilate_kernel_size)
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.erode_kernel_size)
        self.skip_adaptive_threshold = skip_adaptive_threshold
    def _process_to_b(self, image):
        """处理图像到二值图"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        #区域自适应阈值处理
        if self.skip_adaptive_threshold:
            #如果跳过自适应阈值处理，则直接使用otsu二值化
            _, binary = cv2.threshold(gray, 0, 255, self.polar + cv2.THRESH_OTSU)
            # cv2.imshow("binary", binary)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return binary, gray
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, self.polar, 11, 2)
        return binary, gray
    
    def _get_and_process_contours(self, canny_img):
        """获取轮廓并处理,返回轮廓，质心,矩方向,最小外接矩形,bbox,填充完毕的mask,理论单目标，只处理最外层轮廓"""
        contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #获取图像宽高
        height, width = canny_img.shape[:2] 
        mask = np.zeros(canny_img.shape, dtype=np.uint8)
        contours_filtered = []
        for contour in contours:
            if len(contour) < self.min_contour_points:
                continue
            #计算轮廓的最小矩形框，与坐标平行的bbox
            x, y, w, h = cv2.boundingRect(contour)
            #如果这个框的长宽与原图相似，则认为框住了整个图像
            if w / width > self.frame_ratio_threshold and h / height > self.frame_ratio_threshold: 
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
        M = cv2.moments(contour)
        if M["m00"] == 0:
            cx, cy = 0, 0
            theta = 0
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
        #转为角度
        theta = math.degrees(theta)
        return contour, center, theta, shape_wh, bbox, mask
    
    def _process_contour(self, contour,gray, x, y, w, h):
        """处理轮廓，返回膨胀腐蚀处理过的canny图像"""
        #框出原图
        cut_image = gray[y:y+h, x:x+w]
        #进行otsu二值化处理，获取分割阈值
        retval,_ = cv2.threshold(cut_image, 0, 255, self.polar + cv2.THRESH_OTSU)
        max_thresh = min(int(retval * self.otsu_multiplier_high), 255)
        min_thresh = max(int(retval * self.otsu_multiplier_low), 0)
        canny_img = cv2.Canny(cut_image, min_thresh, max_thresh)
        #膨胀腐蚀处理
        canny_img = cv2.dilate(canny_img, self.kernel_dilate, iterations=1)
        canny_img = cv2.erode(canny_img, self.kernel_erode, iterations=1)
        #贴回原图
        return canny_img
    def _empty_result(self):
        """返回空结果"""
        return {
            "contours": [], "centers": [], "thetas": [], 
            "shapes": [], "bboxes": [], 
            "mask": np.zeros((1, 1), dtype=np.uint8)
        }
    def get_contours_image(self, image):
        try:
            binary, gray = self._process_to_b(image)
        except Exception as e:
            print(f"Error processing image to binary: {e}")
            return self._empty_result()
        """返回筛选过的bbox，最小轮廓的外接矩形，膨胀腐蚀处理过的canny全幅图像，根据canny抽取的所有轮廓点"""
        contours, _ = cv2.findContours(binary, self.outer, cv2.CHAIN_APPROX_SIMPLE)
        height, width = binary.shape[:2]
        mask_img = np.zeros(binary.shape, dtype=np.uint8)
        contours_list = []
        center_list = []
        theta_list = []
        shape_list = []
        bbox_list = []
        temp_mask = np.zeros(gray.shape, np.uint8)
        for original_contour in contours:
            if len(original_contour) < self.min_contour_points:
                continue
            #计算轮廓的最小矩形框，与坐标平行的bbox
            x, y, w, h = cv2.boundingRect(original_contour)
            #如果这个框的长宽与原图相似，则认为框住了整个图像
            if w / width > self.frame_ratio_threshold and h / height > self.frame_ratio_threshold: 
                continue
            #框出原图
            temp_mask.fill(0)  # 清空临时mask
            canny_img = self._process_contour(original_contour, gray, x, y, w, h)
            temp_mask[y:y+h, x:x+w] = canny_img
            processed_contour, center, theta, shape_wh, bbox, result_mask = self._get_and_process_contours(temp_mask)
            if processed_contour is None:
                continue
            contours_list.append(processed_contour)
            center_list.append(center)
            theta_list.append(theta)
            shape_list.append(shape_wh)
            bbox_list.append(bbox)
            #将mask添加到mask_img中,单通道
            cv2.drawContours(mask_img, [processed_contour], -1, 255, thickness=cv2.FILLED)
        
        return {"contours": contours_list, "centers": center_list, "thetas": theta_list, "shapes": shape_list, "bboxes": bbox_list, "mask": mask_img}