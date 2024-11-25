from config import TrainSVMOptions

import click
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib
from pathlib import Path
import numpy as np

@click.command()
@click.option("--comment", type=str, default="", show_default=False)
@click.option("--model_base_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--image_size", type=int, default=224, show_default=True)
def train(**kwargs):
    opts = TrainSVMOptions().parse(kwargs, present=True)

    # 加载特征和标签
    features = np.load(Path(kwargs['dataset_path']) / 'flowers-17' /'feature_for_svm.npy')
    labels = np.load(Path(kwargs['dataset_path']) / 'flowers-17' /'label_for_svm.npy')
    key = np.load(Path(kwargs['dataset_path']) / 'flowers-17' /'key_for_svm.npy')
    
    # 为第一种花和第二种花创建标签掩码
    mask_back1 = (key <=1280) & (labels == 0)
    mask_back2 = (key > 1280) & (labels == 0)
    mask_flower1 = labels == 1
    mask_flower2 = labels == 2

    # 分别提取每种的特征
    features_back1 = features[mask_back1]
    features_back2 = features[mask_back2]
    features_flower1 = features[mask_flower1]
    features_flower2 = features[mask_flower2]

    # 创建第一种花和背景的标签
    labels_flower1 = np.ones(features_flower1.shape[0], dtype=int)  # 第一种花的标签为 1
    labels_background1 = np.zeros(features_back1.shape[0], dtype=int)  # 背景的标签为 0

    # 创建第二种花和背景的标签
    labels_flower2 = np.ones(features_flower2.shape[0], dtype=int)  # 第二种花的标签为 1
    labels_background2 = np.zeros(features_back2.shape[0], dtype=int)  # 背景的标签为 0

    # 合并背景和花的特征以及标签，以便进行二分类
    features_train1 = np.vstack((features_flower1, features_back1))
    labels_train1 = np.hstack((labels_flower1, labels_background1))
    features_train2 = np.vstack((features_flower2, features_back2))
    labels_train2 = np.hstack((labels_flower2, labels_background2))

    # 分割数据集为训练集和验证集
    X_train1, X_val1, y_train1, y_val1 = train_test_split(features_train1, labels_train1, test_size=0.3, random_state=42)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(features_train2, labels_train2, test_size=0.3, random_state=42)

    # 训练两个SVM分类器
    svm_classifier1 = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)
    svm_classifier2 = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)
    svm_classifier1.fit(X_train1, y_train1)
    svm_classifier2.fit(X_train2, y_train2)

    # 验证SVM分类器
    y_pred1 = svm_classifier1.predict(X_val1)
    y_pred2 = svm_classifier2.predict(X_val2)
    print(f'Validation accuracy for flower 1: {accuracy_score(y_val1, y_pred1)}')
    print(f'Validation accuracy for flower 2: {accuracy_score(y_val2, y_pred2)}')

    # 保存SVM模型
    folder = Path(opts.model_base_path) / 'checkpoints'
    folder.mkdir()
    joblib.dump(svm_classifier1, folder / "1_svm.pkl")
    joblib.dump(svm_classifier2, folder / "2_svm.pkl")

if __name__ == "__main__":
    train()