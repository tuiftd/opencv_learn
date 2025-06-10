import cv2
import numpy as np
import math
import os
import json
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from toolbox import EdgeProcessorTool as Tool
from class_get_eigenvector import GetImageEigenvector as GetEigenvector



def get_eigenvector(image,edge_processor_obj=None):
    # img_path = r"model_img\moban_1.jpg"
    # # 读取图像
    # image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #添加一圈0
    #gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # _,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 实例化边缘处理工具
    if edge_processor_obj is None:
        edge_processor = Tool(polar="dark", outer=True, skip_adaptive_threshold=True)
    else:
        edge_processor = edge_processor_obj
    message_dict = edge_processor.get_contours_image(gray)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour in message_dict["contours"]:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    #  
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    moban_dict = {
        "mask": mask,
        "contour": message_dict["contours"][0],  # 只取第一个轮廓
        "bbox": message_dict["bboxes"][0],  # 只取第一个bbox
        "shape": message_dict["shapes"][0],
        "theta": message_dict["thetas"][0],
        "center": message_dict["centers"][0]
    }
    # 实例化特征提取类
    eigenvector_extractor = GetEigenvector(moban_dict)
    feature_vector = eigenvector_extractor()
    return feature_vector
def get_moban_eigenvector(img_dir):
    """遍历指定目录下的所有图像文件，获取每个图像的特征向量"""
    eigenvectors = []
    labels = []
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                eigenvector = get_eigenvector(img)
                eigenvectors.append(eigenvector)
                labels.append(filename)
    return {"eigenvectors": eigenvectors, "labels": labels}

def train_knn_classifier(eigenvectors_list, labels_list):
    """训练KNN分类器"""
    knn = KNN(n_neighbors=1,p=1,weights='distance') 
    knn.fit(eigenvectors_list, labels_list)
    return knn

def display_comparison(target_image, predicted_template_name, template_dir):
    """显示目标图像和预测的模板图像的对比"""
    # 读取预测的模板图像
    template_path = os.path.join(template_dir, predicted_template_name)
    template_image = cv2.imread(template_path)
    
    if template_image is None:
        print(f"无法读取模板图像: {template_path}")
        return
    
    # 调整图像大小使其一致（可选）
    target_height, target_width = target_image.shape[:2]
    template_height, template_width = template_image.shape[:2]
    
    # 统一高度，按比例调整宽度
    unified_height = max(target_height, template_height)
    
    # 调整目标图像大小
    target_aspect = target_width / target_height
    new_target_width = int(unified_height * target_aspect)
    target_resized = cv2.resize(target_image, (new_target_width, unified_height))
    
    # 调整模板图像大小
    template_aspect = template_width / template_height
    new_template_width = int(unified_height * template_aspect)
    template_resized = cv2.resize(template_image, (new_template_width, unified_height))
    
    # 水平拼接两张图像
    combined_image = np.hstack((target_resized, template_resized))
    
    # 添加文字标签
    cv2.putText(combined_image, "Target", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_image, "Template", 
                (new_target_width + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 显示结果
    cv2.imshow("KNN Classification Result", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main():
    img_dir = r"c1"
    # 获取模板图像的特征向量
    moban_data = get_moban_eigenvector(img_dir)
    eigenvectors = moban_data["eigenvectors"]
    labels = moban_data["labels"]
    for i, vec in enumerate(eigenvectors):
        print(f"Eigenvector for {labels[i]}: {vec}")
    # 训练KNN分类器
    knn_classifier = train_knn_classifier(eigenvectors, labels)

    # # 保存模型
    # with open("knn_model.json", "w") as f:
    #     json.dump({"eigenvectors": eigenvectors, "labels": labels}, f)

    print("KNN classifier trained")
    img_path = r"c2\5_37.bmp"
    test_image = cv2.imread(img_path)
    # gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_feature_vector = get_eigenvector(test_image, edge_processor_obj=None)
    print("Test feature vector:", test_feature_vector)
    # 使用KNN分类器进行预测
    predicted_label = knn_classifier.predict([test_feature_vector])
    print("Predicted label:", predicted_label)
    #合并指向的目标和预测出来的模板图像显示
    display_comparison(test_image, predicted_label[0], img_dir)


if __name__ == "__main__":
    main()