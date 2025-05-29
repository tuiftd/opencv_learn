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
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    # _,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 实例化边缘处理工具
    if edge_processor_obj is None:
        edge_processor = Tool(polar="light", outer=True, skip_adaptive_threshold=True)
    else:
        edge_processor = edge_processor_obj
    message_dict = edge_processor.get_contours_image(gray)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour in message_dict["contours"]:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
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
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # 使用你的特征提取算法获取特征向量
                eigenvector = get_eigenvector(img)
                eigenvectors.append(eigenvector)
                labels.append(filename)
    return {"eigenvectors": eigenvectors, "labels": labels}

def train_knn_classifier(eigenvectors_list, labels_list):
    """训练KNN分类器"""
    knn = KNN(n_neighbors=1,p=1,weights='distance')  # 你可以根据需要调整k值
    knn.fit(eigenvectors_list, labels_list)
    return knn


def main():
    img_dir = r"model_img"
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
    img_path = r"model_img\moban_4.jpg"
    test_image = cv2.imread(img_path)
    # gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_feature_vector = get_eigenvector(test_image, edge_processor_obj=None)
    print("Test feature vector:", test_feature_vector)
    # 使用KNN分类器进行预测
    predicted_label = knn_classifier.predict([test_feature_vector])
    print("Predicted label:", predicted_label)
if __name__ == "__main__":
    main()