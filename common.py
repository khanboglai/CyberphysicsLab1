""" Испорты, константы, методы и классы общего назначения для всех модулей. """

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import pandas as pd
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import random
import json



SEED = 42 # фиксируем случайность для повторяемости результатов
# устройство для обучения (GPU, если доступно, иначе CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# пути к данным
DATA_DIR = Path("dataset/dataset2-master/dataset2-master/images") # путь к данным
TRAIN_DIR = DATA_DIR / "TRAIN" # путь к тренировочному набору
TEST_DIR = DATA_DIR / "TEST" # путь к тестовому набору
CLASS_NAMES = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir()) # список классов

# настройки обучения
BATCH_SIZE = 32 # размер батча
IMG_SIZE = 224 # стандартный размер для моделей
VAL_SPLIT = 0.2 # доля валидационного набора
NUM_WORKERS = 2 # количество рабочих потоков для загрузки данных


def set_seed(seed=SEED):
    """ Настройка всех генераторов случайных чисел на одно число """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# трансформы для обучения и тестирования
# Преобразование: изменения размера -> в тензор -> нормализация
# Используются страндартные средние и отклонения ImageNet
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
# Преобразование: изменения размера -> в тензор -> нормализация
# Используются страндартные средние и отклонения ImageNet
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


# Dataset
class BloodCellsDataset(Dataset):
    """ Кастомный Dataset для загрузки изображений и меток классов. """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # список пар (путь к изображению, метка класса)
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()]) # список классов
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)} # словарь классов в индексы

        for cls in self.class_names: # проходим по всем классам
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob("*.jpeg"): # проходим по всем изображениям в классе
                self.samples.append((img_path, self.class_to_idx[cls])) # добавляем пару (путь к изображению, метка класса) в список

    def __len__(self):
        return len(self.samples) # возвращаем количество образцов в датасете
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx] # получаем путь к изображению и метку класса
        image = Image.open(img_path).convert("RGB") # открываем изображение и конвертируем в RGB
        if self.transform: # если есть трансформ, применяем его
            image = self.transform(image) # применяем трансформ к изображению
        return image, label # возвращаем изображение и метку класса
    


# функции обучения
def train_epoch(model, loader, criterion, optimizer, device):
    """ Обучение модели на одной эпохе """
    model.train() # режим обучения
    running_loss = 0.0 # начальная потеря
    all_preds = [] # список предсказаний
    all_labels = [] # список меток

    for images, labels in loader: # проходим по всем образцам в loader
        images, labels = images.to(device), labels.to(device) # перемещаем данные на устройство
        optimizer.zero_grad() # обнуляем градиенты
        outputs = model(images) # получаем предсказания
        loss = criterion(outputs, labels) # вычисляем потерю
        loss.backward() # вычисляем градиенты
        optimizer.step() # обновляем параметры
        running_loss += loss.item() * images.size(0) # добавляем потерю к общей потере
        _, preds = torch.max(outputs, 1) # получаем индекс максимального значения
        all_preds.extend(preds.cpu().numpy()) # добавляем предсказания в список
        all_labels.extend(labels.cpu().numpy()) # добавляем метки в список

    avg_loss = running_loss / len(loader.dataset) # вычисляем среднюю потерю
    acc = accuracy_score(all_labels, all_preds) # вычисляем точность
    return avg_loss, acc, all_preds, all_labels # возвращаем среднюю потерю, точность, список предсказаний и список меток


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """ Оценка модели на тестовом наборе """
    model.eval() # режим оценки
    running_loss = 0.0 # начальная потеря
    all_preds = [] # список предсказаний
    all_labels = [] # список меток

    for images, labels in loader: # проходим по всем образцам в loader
        images, labels = images.to(device), labels.to(device) # перемещаем данные на устройство
        outputs = model(images) # получаем предсказания
        loss = criterion(outputs, labels) # вычисляем потерю

        running_loss += loss.item() * images.size(0) # добавляем потерю к общей потере
        _, preds = torch.max(outputs, 1) # получаем индекс максимального значения
        all_preds.extend(preds.cpu().numpy()) # добавляем предсказания в список
        all_labels.extend(labels.cpu().numpy()) # добавляем метки в список
    
    avg_loss = running_loss / len(loader.dataset) # вычисляем среднюю потерю
    acc = accuracy_score(all_labels, all_preds) # вычисляем точность
    bal_acc = balanced_accuracy_score(all_labels, all_preds) # вычисляем сбалансированную точность
    f1 = f1_score(all_labels, all_preds, average="macro") # вычисляем F1 score
    return avg_loss, acc, bal_acc, f1, all_preds, all_labels # возвращаем среднюю потерю, точность, сбалансированную точность, F1 score, список предсказаний и список меток



def train_model(model, model_name, train_loader, val_loader, test_loader, epochs=10, lr=1e-3, device=DEVICE, use_amp=True):
    """ Функция для обучения модели """
    set_seed(SEED)
    model = model.to(device) # перемещаем модель на устройство
    criterion = nn.CrossEntropyLoss() # функция потерь
    optimizer = optim.Adam(model.parameters(), lr=lr) # оптимизатор
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # scheduler

    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None # автоматическое масштабирование градиентов

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "train_acc": []} # словарь для хранения истории обучения

    for epoch in range(epochs): # проходим по всем эпохам
        # --- TRAIN ---
        model.train() # режим обучения
        running_loss = 0.0
        all_preds = [] # список предсказаний
        all_labels = [] # список меток

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # перемещаем данные на устройство
            optimizer.zero_grad() # обнуляем градиенты

            if scaler: # если используется автоматический масштабирование градиентов
                with torch.amp.autocast('cuda'): # режим автоматического масштабирования градиентов
                    outputs = model(images) # получаем предсказания
                    loss = criterion(outputs, labels) # вычисляем потерю
                scaler.scale(loss).backward() # вычисляем градиенты
                scaler.step(optimizer) # обновляем параметры
                scaler.update() # обновляем масштаб градиентов
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1) # получаем индекс максимального значения
            all_preds.extend(preds.cpu().numpy()) # добавляем предсказания в список
            all_labels.extend(labels.cpu().numpy()) # добавляем метки в список

        train_loss = running_loss / len(train_loader.dataset) # вычисляем среднюю потерю
        train_acc = accuracy_score(all_labels, all_preds) # вычисляем точность

        # --- VAL ---
        val_loss, val_acc, val_bal_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device) # оценка на валидационном наборе
        scheduler.step() # обновляем scheduler

        history["train_loss"].append(train_loss) # добавляем среднюю потерю на обучающем наборе в историю
        history["val_loss"].append(val_loss) # добавляем среднюю потерю на валидационном наборе в историю
        history["train_acc"].append(train_acc) # добавляем точность на обучающем наборе в историю
        history["val_acc"].append(val_acc) # добавляем точность на валидационном наборе в историю

        print(f"Epoch {epoch+1}/{epochs} - " # выводим номер эпохи
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - " # выводим среднюю потерю на обучающем наборе и точность
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_bal_acc:.4f}, Val F1: {val_f1:.4f}") # выводим среднюю потерю на валидационном наборе и точность
    
    test_loss, test_acc, test_bal_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device) # оценка на тестовом наборе

    # вывод метрик на тестовом наборе
    print(f"\n {model_name} - TEST")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")

    # вывод матрицы ошибок
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # вывод графиков потерь и точности
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_title(f"Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # вывод отчета о классификации
    print(f"\n{model_name} - Classification report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    
    # вывод результатов
    result = {
        "model": model_name,
        "test_acc": test_acc,
        "test_bal_acc": test_bal_acc,
        "test_f1_macro": test_f1,
        "history": history,
        "confusion_matrix": cm.tolist(),
    }
    return model, result


def load_results(path="outputs/experiment_results.pkl"):
    """ Загрузка результатов из файла """
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_results(results, path="outputs/experiment_results.pkl"):
    """ Сохранение результатов в файл """
    os.makedirs("outputs", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Результаты сохранены в {path}")
