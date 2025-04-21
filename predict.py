# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os
import argparse
import cv2
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Конфигурация ---
DEFAULT_MODEL_PATH = "runs/train/drawing_detection_run3/weights/best.pt"  # Путь к обученной модели
DEFAULT_SOURCE = "6.png"  # Изображение для анализа
DEFAULT_CONF = 0.1  # Порог уверенности
DEFAULT_MAX_DET = 100  # Максимальное количество детекций
DEFAULT_IOU = 0.5  # Порог IoU для NMS

# --- Обработка аргументов командной строки ---
parser = argparse.ArgumentParser(description="Анализ изображений с помощью YOLOv8.")
parser.add_argument(
    '--model',
    type=str,
    default=DEFAULT_MODEL_PATH,
    help=f"Путь к файлу модели (.pt). По умолчанию: {DEFAULT_MODEL_PATH}"
)
parser.add_argument(
    '--source',
    type=str,
    default=DEFAULT_SOURCE,
    help=f"Путь к изображению для анализа. По умолчанию: {DEFAULT_SOURCE}"
)
parser.add_argument(
    '--conf',
    type=float,
    default=DEFAULT_CONF,
    help=f"Порог уверенности для предсказаний. По умолчанию: {DEFAULT_CONF}"
)
parser.add_argument(
    '--max_det',
    type=int,
    default=DEFAULT_MAX_DET,
    help=f"Максимальное количество детекций. По умолчанию: {DEFAULT_MAX_DET}"
)
parser.add_argument(
    '--iou',
    type=float,
    default=DEFAULT_IOU,
    help=f"Порог IoU для Non-Maximum Suppression. По умолчанию: {DEFAULT_IOU}"
)

args = parser.parse_args()

# Используем значения из аргументов
model_path = args.model
source = args.source
conf = args.conf
max_det = args.max_det
iou = args.iou

# --- Определение цветов для классов ---
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
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, size=20)
except Exception as e:
    print(f"Ошибка загрузки шрифта: {e}")
    print("Используем шрифт по умолчанию (может не поддерживать кириллицу).")
    font = ImageFont.load_default()

# --- Загрузка модели ---
print(f"Загрузка модели: {model_path}")
try:
    model = YOLO(model_path)
    print(f"Модель {model_path} успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели {model_path}: {e}")
    exit()

# --- Выполнение предсказания ---
print(f"\nВыполнение анализа на '{source}' с conf={conf}")
try:
    # Проверяем, является ли источник файлом изображения
    if os.path.isfile(source) and source.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = source
    else:
        print(f"Ошибка: Источник '{source}' не является изображением.")
        exit()

    # Выполняем предсказание
    results = model.predict(
        source=img_path,
        conf=conf,
        max_det=max_det,
        iou=iou,
        save=False
    )

    # Визуализация результатов
    for result in results:
        # Загружаем исходное изображение
        img = cv2.imread(img_path)
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение: {img_path}")
            continue

        # Подготовка словаря для группировки
        grouped_objects = {}

        # Получаем имена классов
        classes_names = result.names

        # Конвертируем изображение в RGB для PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Обрабатываем каждую детекцию
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_id = int(cls)
            class_name = classes_names[class_id]
            color = colors[class_id % len(colors)]

            # Группировка объектов
            if class_name not in grouped_objects:
                grouped_objects[class_name] = []
            grouped_objects[class_name].append(box.cpu().numpy().astype(int).tolist())

            # Рисование рамок с помощью PIL
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

            # Формируем метку
            label = f"{class_name} {conf:.2f}"
            label = label.encode('utf-8').decode('utf-8')

            # Рисуем фон и текст
            text_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text((x1, y1 - 30), label, fill=color, font=font)

        # Конвертируем обратно в BGR для сохранения
        img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Сохранение результата
        save_dir = os.path.join("runs/detect", f"exp_{uuid.uuid4().hex[:8]}")
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img_result)
        print(f"Результат сохранен: {output_path}")

        # Вывод группировки
        print("Группировка объектов по классам:")
        for class_name, boxes in grouped_objects.items():
            print(f"{class_name}: {len(boxes)} объектов")

    print("Анализ завершен.")

except Exception as e:
    print(f"Ошибка во время анализа: {e}")

print("\nСкрипт завершил работу.")