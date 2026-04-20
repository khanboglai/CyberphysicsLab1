# Лабораторная работа 1. Киберфизические системы

Студент: Боглаев А.А.

Группа: М8О-406Б-22

## Тема

Проведение исследований с моделями классификации.

## Датасет

Для провердения исследований был выбран следующий датасет: [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data)

12500 изображений

4 вида клеток

Анализ датасета приведен в jupyter [notebook](data_analize.ipynb)

*Важно отметить, что датасет разбит на тренировочный набор и тестовый. Обучение проходит на тренировочном наборе, а проверка проходит на тестовом.*

---

## Метрики качества

- **Accuracy** -- общая доля правильных ответов
- **Balanced Accuracy** -- средняя точность по классам, полезна при дизбалансе
- **Macro-F1** -- среднее F1 по всем классам.

Также для оценок ошибок классификации используется **Confusion Matrix**, а для получения метрик `F1`, `Precision`, `Recall` используется **Classification Report**.

Выбранный набор метрик обеспечивает комплексную оценку: **Accuracy** дает базовое представление об эффективности, а **Balanced Accuracy** и **Macro-F1** страхуют от ложных выводов в случае дисбаланса классов, гарантируя внимание к каждой категории. Использование **Confusion Matrix** и **Classification Report** переводит анализ из плоскости «сухих цифр» к детализации ошибок, позволяя точно определить, какие именно визуальные образы модель путает между собой. Такой подход делает исследование объективным и позволяет увидеть реальное качество работы алгоритма даже на сложных, неоднородных данных.

---

## Инструкция по запуску

1. Склонировать репозиторий

```bash
git clone https://github.com/khanboglai/CyberphysicsLab1.git
```

2. Перейти в папку проекта

```bash
cd CyberphysicsLab1
```

3. Создать виртуальное окружение `venv`

```bash
python3 -m venv venv
```

4. Активировать его


На MacOS или Linux

```bash
source venv/bin/activate
```

На Windows через `cmd`

```bash
venv\Scripts\activate.bat
```

5. Установить зависиомсти из файла [requirements.txt](requirements.txt)

```bash
pip install -r requirements.txt
```

*В файле нужно поменять версию библиотеки torch и torchvision под свои мощности GPU!*

6. Датасет сохранить в папку `dataset` рядом с ноутбуками.

7. Запуск ноутбуков. При запуске настройте ядро на созданное выше вирутальное окружение.

*Запускать ноутбуки необходимо в следующем порядке:*


| №   | Ноутбук                                                |
| --- | ------------------------------------------------------ |
| 1   | [data_analize.ipynb](data_analize.ipynb)               |
| 2   | [baseline.ipynb](baseline.ipynb)                       |
| 3   | [hypothesis_1.ipynb](hypothesis_1.ipynb)               |
| 4   | [hypothesis_2.ipynb](hypothesis_2.ipynb)               |
| 5   | [hypothesis_3.ipynb](hypothesis_3.ipynb)               |
| 6   | [hypothesis_4.ipynb](hypothesis_4.ipynb)               |
| 7   | [hypothesis_5.ipynb](hypothesis_5.ipynb)               |
| 8   | [hypothesis_6.ipynb](hypothesis_6.ipynb)               |
| 9   | [hypothesis_6_swin.ipynb](hypothesis_6_swin.ipynb)     |
| 10  | [custom_cnn_baseline.ipynb](custom_cnn_baseline.ipynb) |
| 11  | [custom_cnn_improved.ipynb](custom_cnn_improved.ipynb) |
| 12  | [custom_vit_baseline.ipynb](custom_vit_baseline.ipynb) |
| 13  | [custom_vit_improved.ipynb](custom_vit_improved.ipynb) |


8. В папку outputs сохраняться веса обученных моделей, а также результаты обучения (метрики, матрица ошибок) в файле `experiment_results.pkl`.

---

## Модели

Для исследования были выбраны модели из `torchvision`: `ResNet18` и `Swin-T`. Модели подбирадись с учетом имеющихся у студента ресурсов.

Обучение baseline моделей в [baseline.ipynb](baseline.ipynb)

---

## Гипотезы

**1.** Аугментация данных улучшает качество модели ([hypothesis_1.ipynb](hypothesis_1.ipynb)).

**2.** Замена модели `StepLR` на `CosineAnnealingLR` улучшит качество модели. ([hypothesis_2.ipynb](hypothesis_2.ipynb))

**3.** Раздельный `learning rate` для backbone и head улучшит качество модели. ([hypothesis_3.ipynb](hypothesis_3.ipynb))

**4.** Увеличение количества эпох обучения улучшит качество модели. ([hypothesis_4.ipynb](hypothesis_4.ipynb))

**5.** Заморозка `backbone` и обучение только `head` с `weight_decay` улучшит качество модели. ([hypothesis_5.ipynb](hypothesis_5.ipynb))

**6.** Комбинация: аугментация + больше эпох + `CosineAnnealingLR` + `label smoothing` улучшит качество модели `ResNet18` ([hypothesis_6.ipynb](hypothesis_6.ipynb)). Для `Swin-T` проверим аугментацию + больше эпох + `CosineAnnealingLR` + `label smoothing` + `AdamW` + `weight_decay` ([hypothesis_6_swin.ipynb](hypothesis_6_swin.ipynb)).

## Результаты готовых моделей

### Общая таблица


| Ключ в experiment_results | Модель                   | Test Accuracy | Test Balanced Accuracy | Test Macro-F1 |
| ------------------------- | ------------------------ | ------------- | ---------------------- | ------------- |
| baseline_resnet18         | ResNet18                 | 0.8544        | 0.8545                 | 0.8588        |
| baseline_swin             | Swin-T                   | 0.8802        | 0.8801                 | 0.8830        |
| resnet18_aug              | ResNet18 + Augmentation  | 0.8741        | 0.8741                 | 0.8760        |
| swin_aug                  | Swin_T + AUG             | 0.8786        | 0.8785                 | 0.8814        |
| resnet18_cosine           | ResNet18_Cosine          | 0.8709        | 0.8709                 | 0.8745        |
| swin_cosine               | Swin-T_Cosine            | 0.8806        | 0.8805                 | 0.8834        |
| resnet18_split            | ResNet18_Split           | 0.8585        | 0.8585                 | 0.8622        |
| swin_split                | Swin-T_Split             | 0.8762        | 0.8761                 | 0.8789        |
| resnet18_es               | ResNet18_ES              | 0.8701        | 0.8701                 | 0.8731        |
| swin_es                   | Swin-T_ES                | 0.8810        | 0.8809                 | 0.8840        |
| resnet18_frozen           | ResNet18_Frozen          | 0.6441        | 0.6443                 | 0.6402        |
| swin_frozen               | Swin-T_Frozen            | 0.7125        | 0.7123                 | 0.7125        |
| ResNet18_Aug+Smooth+30ep  | ResNet18_Aug+Smooth+30ep | 0.8938        | 0.8938                 | 0.8958        |
| Swin_T_adamW_warmup       | Swin_T_adamW_warmup      | 0.8938        | 0.8938                 | 0.8957        |


### Baseline + Top-1 для каждой архитектуры


| Семейство | Тип      | Ключ в experiment_results | Модель                   | Test Accuracy | Test Balanced Accuracy | Test Macro-F1 |
| --------- | -------- | ------------------------- | ------------------------ | ------------- | ---------------------- | ------------- |
| ResNet18  | Baseline | baseline_resnet18         | ResNet18                 | 0.8544        | 0.8545                 | 0.8588        |
| ResNet18  | Top-1    | ResNet18_Aug+Smooth+30ep  | ResNet18_Aug+Smooth+30ep | 0.8938        | 0.8938                 | 0.8958        |
| Swin-T    | Baseline | baseline_swin             | Swin-T                   | 0.8802        | 0.8801                 | 0.8830        |
| Swin-T    | Top-1    | Swin_T_adamW_warmup       | Swin_T_adamW_warmup      | 0.8938        | 0.8938                 | 0.8957        |


---

## Улучшенный baseline

Для модели ResNet18: Аугментация с lable smoohing и CosineAnnealingLR с обучением на 30 эпохах

Для модели Swin-T: Аугментация с CosineAnnealingLR, AdamW и Warmup с обучением на 30 эпохах.

---

## Результаты кастомных реализаций

Модели реализованы самостоятельно с нуля (без `torchvision` архитектур).

- **CustomMiniResNet** — ResNet-подобная CNN: `ConvBlock(3→32)` + 4×`ResidualBlock` (stride=2) + `AdaptiveAvgPool` + `FC(512→4)`
- **CustomViT** — Vision Transformer: `PatchEmbed(16×16, d=192)` + `[CLS]` + `LearnablePosEmbed` + 4×`TransformerBlock(heads=3, mlp=768)` + `FC(192→4)`

### Кастомные модели — общая таблица


| Ноутбук                                                | Модель           | Техники                                                 | Test Accuracy | Test Balanced Accuracy | Test Macro-F1 |
| ------------------------------------------------------ | ---------------- | ------------------------------------------------------- | ------------- | ---------------------- | ------------- |
| [custom_cnn_baseline.ipynb](custom_cnn_baseline.ipynb) | CustomMiniResNet | Adam + StepLR, 10 эп, без аугм.                         | 0.8762        | 0.8761                 | 0.8792        |
| [custom_cnn_improved.ipynb](custom_cnn_improved.ipynb) | CustomMiniResNet | Adam + CosineAnnealingLR + LabelSmoothing + Aug, 30 эп  | 0.8822        | 0.8821                 | 0.8851        |
| [custom_vit_baseline.ipynb](custom_vit_baseline.ipynb) | CustomViT        | Adam + StepLR, 10 эп, без аугм.                         | 0.8291        | 0.8292                 | 0.8289        |
| [custom_vit_improved.ipynb](custom_vit_improved.ipynb) | CustomViT        | AdamW + CosineAnnealingLR + LabelSmoothing + Aug, 30 эп | 0.8552        | 0.8553                 | 0.8556        |


### Сравнение кастомных моделей с готовыми архитектурами


| Семейство   | Тип                  | Модель                    | Test Accuracy | Test Balanced Accuracy | Test Macro-F1 |
| ----------- | -------------------- | ------------------------- | ------------- | ---------------------- | ------------- |
| CNN         | Baseline (готовая)   | ResNet18                  | 0.8544        | 0.8545                 | 0.8588        |
| CNN         | Baseline (кастомная) | CustomMiniResNet baseline | 0.8762        | 0.8761                 | 0.8792        |
| CNN         | Improved (готовая)   | ResNet18_Aug+Smooth+30ep  | 0.8938        | 0.8938                 | 0.8958        |
| CNN         | Improved (кастомная) | CustomMiniResNet improved | 0.8822        | 0.8821                 | 0.8851        |
| Transformer | Baseline (готовая)   | Swin-T                    | 0.8802        | 0.8801                 | 0.8830        |
| Transformer | Baseline (кастомная) | CustomViT baseline        | 0.8291        | 0.8292                 | 0.8289        |
| Transformer | Improved (готовая)   | Swin_T_adamW_warmup       | 0.8938        | 0.8938                 | 0.8957        |
| Transformer | Improved (кастомная) | CustomViT improved        | 0.8552        | 0.8553                 | 0.8556        |


### Вывод

Классификация клеток крови — задача с выраженными локальными визуальными признаками: форма ядра, текстура цитоплазмы, характерное окрашивание. Свёрточные сети по природе своей устроены именно для таких признаков — каждый фильтр ищет локальный паттерн, а иерархия слоёв постепенно строит представление от краёв к форме и к классу. Кастомная `CustomMiniResNet` с помощью остаточных соединений эффективно обучается даже на ~8 тыс. изображений, достигая 0.8762 в baseline и 0.8822 после улучшений. Это сопоставимо с предобученным ResNet18 0.8544 и лишь на ~1.2% хуже лучшей конфигурации `ResNet18_Aug+Smooth+30ep` (0.8938).

Трансформеры полагаются на анализ внутренних взаимосвязей, который строит глобальные зависимости между патчами изображения. Это мощный инструмент, но у него есть фундаментальное ограничение: для того чтобы attention-матрицы стали осмысленными, модели нужно увидеть очень большое количество примеров. На небольшом датасете (~8 тыс. train) кастомный `CustomViT` практически не имеет шансов самостоятельно выучить, какие патчи изображения важны — он не успевает заполнить полезным содержанием 197×197 позиционных взаимодействий за 10 эпох. Baseline даёт 0.8291, после применения улучшений — 0.8552: прирост есть, но отставание от кастомной CNN (0.8762/0.8822) остаётся существенным. Готовый Swin-T (0.8802–0.8938) при этом работает значительно лучше, потому что использует предобучение на ImageNet, то есть его attention-блоки уже настроены на осмысленные визуальные паттерны.

**Итог.** Для задач с небольшим датасетом и локально-различимыми классами CNN является предпочтительной архитектурой при обучении с нуля: она проще, быстрее сходится и не требует предобучения для достижения конкурентного качества. Трансформеры раскрывают потенциал только с предобучением или на значительно больших объёмах данных. Также важно учитывать качество датасета, на котором будет обучаться модель. Особенно важно заранее анализировать тестовый набор, чтобы понять, чем он отличается от обучающего. Часто получается, что модель идеально запомнила обучающие данные, но на тестовых может показывать более низкое качество, так как тестовый набор был сделан при других условиях.