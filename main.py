# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os
import argparse
import cv2  # Для обработки изображений
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # Для работы с текстом и шрифтами

# --- Конфигурация ---
# Значения по умолчанию, могут быть переопределены аргументами командной строки
DEFAULT_DATA_CONFIG = "dataset/data.yaml"
DEFAULT_MODEL_TO_TRAIN = 'runs/train/drawing_detection_run3/weights/best.pt'  # Модель по умолчанию для начала обучения
DEFAULT_EPOCHS = 100
DEFAULT_IMG_SIZE = 1088
DEFAULT_BATCH_SIZE = 8
DEFAULT_RUN_NAME = 'drawing_detection_run'
DEFAULT_PROJECT_NAME = 'runs/train'
DEFAULT_PREDICTION_SOURCE = '6.png'
DEFAULT_PREDICTION_CONF = 0.2  # Порог уверенности для предсказаний
DEFAULT_SAVE_PERIOD = 10
DEFAULT_MAX_DET = 100  # Ограничиваем максимальное количество детекций
DEFAULT_IOU = 0.5  # Порог IoU для Non-Maximum Suppression

# --- Обработка аргументов командной строки ---
parser = argparse.ArgumentParser(description="Обучение и использование модели YOLOv8 для детекции.")
parser.add_argument(
    '--model',
    type=str,
    default=DEFAULT_MODEL_TO_TRAIN,
    help=f"Путь к файлу модели (.pt) или конфигурации (.yaml) для начала обучения или дообучения. По умолчанию: {DEFAULT_MODEL_TO_TRAIN}"
)
parser.add_argument(
    '--data',
    type=str,
    default=DEFAULT_DATA_CONFIG,
    help=f"Путь к файлу конфигурации датасета (data.yaml). По умолчанию: {DEFAULT_DATA_CONFIG}"
)
parser.add_argument(
    '--epochs',
    type=int,
    default=DEFAULT_EPOCHS,
    help=f"Количество эпох обучения. По умолчанию: {DEFAULT_EPOCHS}"
)
parser.add_argument(
    '--imgsz',
    type=int,
    default=DEFAULT_IMG_SIZE,
    help=f"Размер изображения для обучения. По умолчанию: {DEFAULT_IMG_SIZE}"
)
parser.add_argument(
    '--batch',
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help=f"Размер батча. Уменьшите, если не хватает видеопамяти. По умолчанию: {DEFAULT_BATCH_SIZE}"
)
parser.add_argument(
    '--name',
    type=str,
    default=DEFAULT_RUN_NAME,
    help=f"Имя папки для сохранения результатов этого запуска. По умолчанию: {DEFAULT_RUN_NAME}"
)
parser.add_argument(
    '--project',
    type=str,
    default=DEFAULT_PROJECT_NAME,
    help=f"Общая папка для всех запусков обучения. По умолчанию: {DEFAULT_PROJECT_NAME}"
)
parser.add_argument(
    '--save_period',
    type=int,
    default=DEFAULT_SAVE_PERIOD,
    help=f"Сохранять чекпоинт каждые N эпох (-1 для сохранения только best/last). По умолчанию: {DEFAULT_SAVE_PERIOD}"
)
parser.add_argument(
    '--predict_source',
    type=str,
    default=DEFAULT_PREDICTION_SOURCE,
    help=f"Изображение или папка для теста после обучения. По умолчанию: {DEFAULT_PREDICTION_SOURCE}"
)
parser.add_argument(
    '--predict_conf',
    type=float,
    default=DEFAULT_PREDICTION_CONF,
    help=f"Порог уверенности для предсказаний. По умолчанию: {DEFAULT_PREDICTION_CONF}"
)
parser.add_argument(
    '--max_det',
    type=int,
    default=DEFAULT_MAX_DET,
    help=f"Максимальное количество детекций на изображении. По умолчанию: {DEFAULT_MAX_DET}"
)
parser.add_argument(
    '--iou',
    type=float,
    default=DEFAULT_IOU,
    help=f"Порог IoU для Non-Maximum Suppression. По умолчанию: {DEFAULT_IOU}"
)
parser.add_argument(
    '--exist_ok',
    action='store_true',  # Если флаг указан, значение будет True
    help="Разрешить перезапись существующего запуска с тем же именем в проекте."
)

args = parser.parse_args()

# Используем значения из аргументов или значения по умолчанию
data_config_path = args.data
model_to_train_path = args.model
num_epochs = args.epochs
image_size = args.imgsz
batch_size = args.batch
run_name = args.name
project_name = args.project
save_period = args.save_period
prediction_source = args.predict_source
prediction_conf = args.predict_conf
max_det = args.max_det
iou = args.iou
exist_ok_flag = args.exist_ok

# --- Определение цветов для классов (темные цвета, контрастные на белом фоне) ---
colors = [
    (0, 0, 128),    # Темно-синий
    (0, 128, 0),    # Темно-зеленый
    (128, 0, 0),    # Темно-красный
    (128, 0, 128),  # Темно-фиолетовый
    (139, 69, 19),  # Темно-коричневый
    (64, 64, 64),   # Темно-серый
]

# --- Загрузка шрифта для PIL с поддержкой кириллицы ---
try:
    # Путь к шрифту DejaVuSans, подходящий для WSL2
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, size=20)  # Размер шрифта 20
except Exception as e:
    print(f"Ошибка загрузки шрифта: {e}")
    print("Используем шрифт по умолчанию (может не поддерживать кириллицу).")
    font = ImageFont.load_default()

# --- 1. Загрузка модели для обучения ---
print(f"Загрузка модели для начала обучения (или дообучения): {model_to_train_path}")
if not os.path.exists(model_to_train_path):
    # Проверяем, является ли это стандартным именем модели ultralytics (например, 'yolov8n.pt')
    if not model_to_train_path in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8y.pt', 'yolov8x.pt',
                                   'yolov8n.yaml', 'yolov8s.yaml', 'yolov8m.yaml', 'yolov8l.yaml', 'yolov8x.yaml']:
        print(f"Ошибка: Файл модели {model_to_train_path} не найден!")
        exit()
    else:
        print(f"Используется стандартная модель/конфигурация Ultralytics: {model_to_train_path}. Загрузка будет выполнена библиотекой.")

# Загружаем модель, указанную пользователем
try:
    model = YOLO(model_to_train_path)
    print(f"Модель {model_to_train_path} успешно инициализирована.")
except Exception as e:
    print(f"Ошибка при загрузке модели {model_to_train_path}: {e}")
    exit()

# --- 2. Проверка файла конфигурации данных ---
print(f"\nПроверка конфигурации датасета: {data_config_path}")
if not os.path.exists(data_config_path):
    print(f"Ошибка: Файл конфигурации датасета {data_config_path} не найден!")
    exit()

print("Содержимое файла конфигурации:")
try:
    with open(data_config_path, 'r', encoding='utf-8') as f:
        print(f.read())
except Exception as e:
    print(f"Ошибка чтения файла конфигурации: {e}")
    exit()

# --- 3. Обучение модели ---
print("\n--- Начало обучения модели ---")
print(f"Стартовая модель: {model_to_train_path}")
print(f"Датасет: {data_config_path}")
print(f"Эпохи: {num_epochs}")
print(f"Размер изображения: {image_size}")
print(f"Размер батча: {batch_size}")
print(f"Имя запуска: {run_name}")
print(f"Проект: {project_name}")
print(f"Период сохранения: {save_period}")
print(f"Разрешена перезапись: {exist_ok_flag}")

# Запускаем обучение
try:
    results_train = model.train(
        data=data_config_path,
        epochs=num_epochs,
        imgsz=image_size,
        batch=batch_size,
        name=run_name,
        project=project_name,
        save=True,
        save_period=save_period,
        exist_ok=exist_ok_flag,
    )
except Exception as e:
    print(f"\nОшибка во время обучения: {e}")
    exit()

print("\n--- Обучение завершено ---")

# --- 4. Определение пути к лучшей модели ---
save_dir = results_train.save_dir
best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
last_model_path = os.path.join(save_dir, 'weights', 'last.pt')

print(f"Результаты обучения сохранены в папке: {save_dir}")

# Проверка, что лучшая модель сохранена
if os.path.exists(best_model_path):
    print(f"Лучшая модель (best.pt) сохранена по пути: {best_model_path}")
    model_to_predict_path = best_model_path
elif os.path.exists(last_model_path):
    print(f"Предупреждение: Лучшая модель (best.pt) не найдена. Используем последнюю модель (last.pt): {last_model_path}")
    model_to_predict_path = last_model_path
else:
    print(f"Ошибка: Ни лучшая (best.pt), ни последняя (last.pt) модель не найдены в {os.path.join(save_dir, 'weights')}")
    print("Невозможно выполнить предсказание.")
    exit()

# --- 5. Загрузка обученной модели и использование для предсказания ---
print("\n--- Загрузка обученной модели для предсказания ---")
try:
    trained_model = YOLO(model_to_predict_path)
    print(f"Модель {model_to_predict_path} успешно загружена для предсказания.")
except Exception as e:
    print(f"Ошибка при загрузке обученной модели {model_to_predict_path}: {e}")
    exit()

print(f"\n--- Выполнение предсказания на '{prediction_source}' с conf={prediction_conf} ---")
try:
    # Проверяем, является ли prediction_source файлом изображения
    if os.path.isfile(prediction_source) and prediction_source.lower().endswith(('.png', '.jpg', '.jpeg')):
        source = prediction_source
    else:
        source = prediction_source
        print(f"Источник не является изображением или это папка: {prediction_source}. Обработка без изменений.")

    # Выполняем предсказание
    results_predict = trained_model.predict(
        source=source,
        conf=prediction_conf,
        max_det=max_det,  # Ограничиваем количество детекций
        iou=iou,  # Устанавливаем порог NMS
        save=False  # Не сохраняем автоматически, будем рисовать рамки вручную
    )

    # Кастомная визуализация результатов
    for result in results_predict:
        # Загружаем исходное изображение
        img = cv2.imread(prediction_source) if os.path.isfile(prediction_source) else result.orig_img
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение для визуализации: {prediction_source}")
            continue

        # Подготовка словаря для группировки результатов по классам
        grouped_objects = {}

        # Получаем имена классов
        classes_names = result.names

        # Конвертируем изображение из BGR (OpenCV) в RGB (для PIL)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Обрабатываем каждую детекцию
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_id = int(cls)
            class_name = classes_names[class_id]
            color = colors[class_id % len(colors)]  # Выбор цвета для класса

            # Группировка объектов по классам
            if class_name not in grouped_objects:
                grouped_objects[class_name] = []
            grouped_objects[class_name].append(box.cpu().numpy().astype(int).tolist())

            # Рисование рамок с помощью PIL
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

            # Формируем метку
            label = f"{class_name} {conf:.2f}"
            label = label.encode('utf-8').decode('utf-8')  # Убедимся, что текст в UTF-8

            # Рисуем фон и текст с помощью PIL
            text_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0))  # Черный фон под текстом
            draw.text((x1, y1 - 30), label, fill=color, font=font)

        # Конвертируем изображение обратно в BGR для OpenCV
        img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Сохранение результата
        save_dir = os.path.join("runs/detect", f"exp_{uuid.uuid4().hex[:8]}")
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, os.path.basename(prediction_source))
        cv2.imwrite(output_path, img_result)
        print(f"Результат предсказания сохранен: {output_path}")

        # Вывод группировки объектов
        print("Группировка объектов по классам:")
        for class_name, boxes in grouped_objects.items():
            print(f"{class_name}: {len(boxes)} объектов")

    print(f"Предсказание завершено.")

except Exception as e:
    print(f"Ошибка во время предсказания: {e}")

print("\nСкрипт завершил работу.")