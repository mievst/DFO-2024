import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import cv2
import json
import numpy as np
import joblib
from datetime import timedelta

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.y = y.copy()
        self.x = x.copy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        sample_y = self.y[idx]
        return torch.tensor(sample).float(), torch.tensor(sample_y).float()


class CNN(nn.Module):
    def __init__(self, num_classes) -> None:
        super(CNN, self).__init__()
        self.cnn = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        num_ftrs = self.cnn.fc.in_features
        #for param in self.cnn.parameters():
            #param.requires_grad = False
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_ftrs, num_ftrs//2)
        self.fc2 = nn.Linear(num_ftrs//2, num_ftrs//4)
        self.fc3 = nn.Linear(num_ftrs//4, num_classes)

    def _get_conv_output_size(self, shape):
        o = self.cnn(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        batch_size, h, w, c = x.size()
        x = x.view(batch_size, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = x.view(-1, 1)
        return x


def get_timestamps(predictions, threshold=0.5, frame_rate=30, window_size=1):
    """
    Получение таймкодов нарушений на основе предсказаний модели.

    predictions: np.array, форма [seq_length]
        Предсказанные вероятности наличия нарушения для каждого окна кадров.
    threshold: float
        Пороговое значение для определения нарушения.
    frame_rate: int
        Количество кадров в секунду.
    window_size: int
        Количество кадров в одном окне.

    Возвращает список кортежей (start_time, end_time) для каждого обнаруженного нарушения.
    """
    timestamps = []
    in_violation = False
    start_time = None

    for i, prob in enumerate(predictions):
        if prob >= threshold and not in_violation:
            in_violation = True
            start_time = i * window_size / frame_rate
        elif prob < threshold and in_violation:
            in_violation = False
            end_time = (i + 1) * window_size / frame_rate
            timestamps.append((start_time, end_time))

    # Если нарушение продолжается до конца видео
    if in_violation:
        end_time = len(predictions) * window_size / frame_rate
        timestamps.append((start_time, end_time))

    return timestamps


class ModelManager():
    def __init__(self):
        self.classificators = {}
        model = CNN(num_classes=1)  # Одна выходная нейрона для вероятности нарушения
        model.load_state_dict(torch.load('checkpoint.pt'))
        self.classificators["подлезание"] = model

    def predict(self, video):
        """
        _summary_

        Args:
            video (_type_): кадры видео нужно сжимать до (7,7,3)

        Returns:
            _type_: _description_
        """
        output = {}
        for key in self.classificators.keys():
            pred = self.classificators[key](video)
            output[key] = get_timestamps(pred)
        return output
