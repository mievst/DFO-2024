import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import cv2
import numpy as np
from datetime import timedelta


TARGET_RESOLUTION = (50,50)
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
        num_ftrs = 2048
        for param in self.cnn.parameters():
            param.requires_grad = False
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


class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(64 * 144, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_length, w, h, c = x.size()
        c_in = x.view(batch_size * seq_length, c, h, w)
        #c_in = self.flatten(c_in)
        c_out = self.cnn(c_in)
        c_out = c_out.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(c_out)
        lstm_out = lstm_out[:, -1, :]
        fc_out = self.fc(lstm_out)
        out = self.sigmoid(fc_out)
        return out

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


def seconds_to_time(total_seconds):
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return '{:02d}:{:02d}'.format(minutes, seconds)

class ModelManager():
    def __init__(self):
        self.classificators = {}
        model = CNNLSTM()  # Одна выходная нейрона для вероятности нарушения
        model.load_state_dict(torch.load('./checkpoints/cnnlstm.pt', map_location=torch.device("cpu")))
        self.classificators["Подлезание под вагоны стоящего состава"] = model

    def predict(self, video):
        """
        _summary_

        Args:
            video (_type_): кадры видео нужно сжимать до (7,7,3)

        Returns:
            _type_: _description_
        """
        output = {}
        #video = torch.tensor(video).cpu().float()
        for key in self.classificators.keys():
            pred = []
            inp = self.__cnn_lstm_data_preparation(video)
            for i in inp:
                out = self.classificators[key](i)
                out = out.cpu().detach()[0]
                for j in out:
                    pred.append(j)
            pred = torch.tensor(pred)
            output[key] = get_timestamps(pred)
        return output

    def __cnn_data_preparation(self, data):
        return torch.tensor(data).cpu().float()

    def __cnn_lstm_data_preparation(self, data):
        l = len(data) // 10
        vid = []
        for i in range(l+1):
            batch = []
            for j in range(10):
                if i*10+j < len(data):
                    batch.append(data[i*10+j])
                else:
                    batch.append(np.zeros_like(data[0]))
            vid.append([batch])
        inp = torch.tensor(vid).float()
        inp = inp.cpu()
        return inp

    def predict_in_folder(self, folder_path):
        video_names = os.listdir(folder_path)
        outputs = []
        video_names = [x for x in video_names if str.lower(x).endswith("mp4")]
        for video_name in video_names:
            video_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, TARGET_RESOLUTION)
                frames.append(frame / 255)
                frames.append(frame)

            cap.release()

            # Convert the frames list into a numpy array
            video_array = np.array(frames)
            pred = self.predict(video_array)
            for key, val in pred.items():
                if len(val) > 0:
                    for item in val:
                        output= {
                            "name": video_name,
                            "start": seconds_to_time(item[0]),
                            "end": seconds_to_time(item[1]),
                            "type": key
                        }
                        outputs.append(output)
        return outputs
