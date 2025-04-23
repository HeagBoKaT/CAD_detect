# -*- coding: utf-8 -*-
import os
import glob
import uuid
import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# --- Configuration ---
DEFAULT_DATA_CONFIG = "dataset/data.yaml"
DEFAULT_MODEL_TO_TRAIN = "runs/train/drawing_detection_run2/weights/best.pt"
DEFAULT_EPOCHS = 1000
DEFAULT_IMG_SIZE = 1600
DEFAULT_BATCH_SIZE = 4
DEFAULT_RUN_NAME = "drawing_detection_run"
DEFAULT_PROJECT_NAME = "runs/train"
DEFAULT_PREDICTION_SOURCE = "test"  # Updated to point to the test folder
DEFAULT_PREDICTION_CONF = 0.3
DEFAULT_SAVE_PERIOD = 10
DEFAULT_MAX_DET = 100
DEFAULT_IOU = 0.5
DEFAULT_XML_DIR = "source_data/correct/"
DEFAULT_IMG_DIR = "source_data/correct/"
DEFAULT_OUTPUT_LABELS_DIR = "dataset/labels/train/"

# Class mapping for XML to YOLO conversion
class_mapping = {
    "Общая шероховатость": 0,
    "Технические требования": 1,
    "Линейный размер": 2,
    "Диаметральный размер": 3,
    "Таблица параметров": 4,
    "Угловой размер": 5,
    "Шероховатость": 6,
    "Фаска": 7
}

# Colors for class visualization (expanded to 11 colors)
colors = [
    (0, 0, 128),    # Dark blue
    (0, 128, 0),    # Dark green
    (128, 0, 0),    # Dark red
    (128, 0, 128),  # Dark purple
    (139, 69, 19),  # Dark brown
    (64, 64, 64),   # Dark gray
    (0, 128, 128),  # Teal
    (255, 165, 0),  # Orange
    (128, 128, 0),  # Olive
    (255, 20, 147), # Deep pink
    (70, 130, 180)  # Steel blue
]

# Load font for PIL with Cyrillic support
try:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, size=20)
except Exception as e:
    console.print(f"[red]Ошибка загрузки шрифта: {e}[/red]")
    console.print("[yellow]Используем шрифт по умолчанию (может не поддерживать кириллицу).[/yellow]")
    font = ImageFont.load_default()

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Application with Rich Interface")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_TO_TRAIN, help=f"Path to model (.pt or .yaml). Default: {DEFAULT_MODEL_TO_TRAIN}")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_CONFIG, help=f"Path to dataset config (data.yaml). Default: {DEFAULT_DATA_CONFIG}")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Number of training epochs. Default: {DEFAULT_EPOCHS}")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help=f"Image size for training. Default: {DEFAULT_IMG_SIZE}")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size. Default: {DEFAULT_BATCH_SIZE}")
    parser.add_argument("--name", type=str, default=DEFAULT_RUN_NAME, help=f"Run name for saving results. Default: {DEFAULT_RUN_NAME}")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT_NAME, help=f"Project directory for training runs. Default: {DEFAULT_PROJECT_NAME}")
    parser.add_argument("--save_period", type=int, default=DEFAULT_SAVE_PERIOD, help=f"Save checkpoint every N epochs. Default: {DEFAULT_SAVE_PERIOD}")
    parser.add_argument("--predict_source", type=str, default=DEFAULT_PREDICTION_SOURCE, help=f"Folder for prediction. Default: {DEFAULT_PREDICTION_SOURCE}")
    parser.add_argument("--predict_conf", type=float, default=DEFAULT_PREDICTION_CONF, help=f"Confidence threshold for predictions. Default: {DEFAULT_PREDICTION_CONF}")
    parser.add_argument("--max_det", type=int, default=DEFAULT_MAX_DET, help=f"Max detections per image. Default: {DEFAULT_MAX_DET}")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help=f"IoU threshold for NMS. Default: {DEFAULT_IOU}")
    parser.add_argument("--xml_dir", type=str, default=DEFAULT_XML_DIR, help=f"Directory with XML annotations. Default: {DEFAULT_XML_DIR}")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR, help=f"Directory with images. Default: {DEFAULT_IMG_DIR}")
    parser.add_argument("--output_labels_dir", type=str, default=DEFAULT_OUTPUT_LABELS_DIR, help=f"Output directory for YOLO labels. Default: {DEFAULT_OUTPUT_LABELS_DIR}")
    parser.add_argument("--exist_ok", action="store_true", help="Allow overwriting existing run with same name.")
    return parser.parse_args()

# --- XML to YOLO Conversion ---
def convert_voc_to_yolo(xml_file, img_file, output_dir, img_width=1247, img_height=1760):
    base_filename = os.path.splitext(os.path.basename(xml_file))[0]
    out_txt_path = os.path.join(output_dir, base_filename + ".txt")

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            console.print(f"[yellow]Предупреждение: Тег 'size' не найден в {xml_file}. Используем размеры {img_width}x{img_height}.[/yellow]")
        else:
            img_width = int(size.find("width").text)
            img_height = int(size.find("height").text)

        if img_width <= 0 or img_height <= 0:
            console.print(f"[red]Ошибка: Некорректные размеры изображения ({img_width}x{img_height}) в {xml_file}. Пропускаем.[/red]")
            return

        lines_to_write = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                console.print(f"[yellow]Предупреждение: Неизвестный класс '{class_name}' в {xml_file}. Пропускаем объект.[/yellow]")
                continue

            class_index = class_mapping[class_name]
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            x_center_norm = max(0.0, min(1.0, x_center / img_width))
            y_center_norm = max(0.0, min(1.0, y_center / img_height))
            width_norm = max(0.0, min(1.0, width / img_width))
            height_norm = max(0.0, min(1.0, height / img_height))

            lines_to_write.append(f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

        os.makedirs(output_dir, exist_ok=True)
        if lines_to_write:
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines_to_write))
            console.print(f"[green]Создан файл аннотации: {out_txt_path}[/green]")
        elif os.path.exists(out_txt_path):
            os.remove(out_txt_path)
            console.print(f"[yellow]Удален пустой файл аннотации: {out_txt_path}[/yellow]")

    except ET.ParseError:
        console.print(f"[red]Ошибка: Не удалось разобрать XML файл {xml_file}.[/red]")
    except Exception as e:
        console.print(f"[red]Ошибка при обработке {xml_file}: {e}[/red]")

def convert_xml_to_yolo_batch(args):
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))
    console.print(f"[cyan]Найдено {len(xml_files)} XML файлов.[/cyan]")
    
    if not xml_files:
        console.print(f"[red]XML файлы не найдены в '{args.xml_dir}'. Проверьте путь.[/red]")
        return

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("[cyan]Конвертация XML в YOLO формат...", total=len(xml_files))
        for xml_file in xml_files:
            base_filename = os.path.splitext(os.path.basename(xml_file))[0]
            img_filename = base_filename + ".png"
            img_file_path = os.path.join(args.img_dir, img_filename)

            if not os.path.exists(img_file_path):
                found_img = False
                for ext in [".jpg", ".jpeg", ".bmp"]:
                    img_filename = base_filename + ext
                    img_file_path = os.path.join(args.img_dir, img_filename)
                    if os.path.exists(img_file_path):
                        found_img = True
                        break
                if not found_img:
                    console.print(f"[yellow]Предупреждение: Файл изображения для {xml_file} не найден. Пропускаем.[/yellow]")
                    progress.advance(task)
                    continue

            convert_voc_to_yolo(xml_file, img_file_path, args.output_labels_dir)
            progress.advance(task)
    
    console.print("[green]Конвертация завершена.[/green]")

# --- Model Training ---
def train_model(args):
    console.print(f"[cyan]Загрузка модели для обучения: {args.model}[/cyan]")
    if not os.path.exists(args.model) and args.model not in [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml", "yolov8l.yaml", "yolov8x.yaml"
    ]:
        console.print(f"[red]Ошибка: Файл модели {args.model} не найден![/red]")
        return

    try:
        model = YOLO(args.model)
        console.print(f"[green]Модель {args.model} успешно инициализирована.[/green]")
    except Exception as e:
        console.print(f"[red]Ошибка при загрузке модели {args.model}: {e}[/red]")
        return

    console.print(f"[cyan]Проверка конфигурации датасета: {args.data}[/cyan]")
    if not os.path.exists(args.data):
        console.print(f"[red]Ошибка: Файл конфигурации датасета {args.data} не найден![/red]")
        return

    console.print("[cyan]Содержимое файла конфигурации:[/cyan]")
    try:
        with open(args.data, "r", encoding="utf-8") as f:
            console.print(f"[white]{f.read()}[/white]")
    except Exception as e:
        console.print(f"[red]Ошибка чтения файла конфигурации: {e}[/red]")
        return

    console.print(Panel.fit(
        f"[bold]Параметры обучения:[/bold]\n"
        f"Модель: {args.model}\n"
        f"Датасет: {args.data}\n"
        f"Эпохи: {args.epochs}\n"
        f"Размер изображения: {args.imgsz}\n"
        f"Размер батча: {args.batch}\n"
        f"Имя запуска: {args.name}\n"
        f"Проект: {args.project}\n"
        f"Период сохранения: {args.save_period}\n"
        f"Перезапись: {args.exist_ok}",
        title="Обучение модели",
        border_style="green"
    ))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("[cyan]Обучение модели...", total=args.epochs)
        try:
            results_train = model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                name=args.name,
                project=args.project,
                save=True,
                save_period=args.save_period,
                exist_ok=args.exist_ok,
            )
            progress.advance(task, advance=args.epochs)
        except Exception as e:
            console.print(f"[red]Ошибка во время обучения: {e}[/red]")
            return

    console.print("[green]Обучение завершено.[/green]")
    save_dir = results_train.save_dir
    best_model_path = os.path.join(save_dir, "weights", "best.pt")
    last_model_path = os.path.join(save_dir, "weights", "last.pt")

    console.print(f"[cyan]Результаты обучения сохранены в: {save_dir}[/cyan]")
    if os.path.exists(best_model_path):
        console.print(f"[green]Лучшая модель сохранена: {best_model_path}[/green]")
    elif os.path.exists(last_model_path):
        console.print(f"[yellow]Предупреждение: Лучшая модель не найдена. Последняя модель: {last_model_path}[/yellow]")
    else:
        console.print(f"[red]Ошибка: Модели не найдены в {os.path.join(save_dir, 'weights')}[/red]")

# --- Prediction ---
def predict_image(args):
    console.print(f"[cyan]Загрузка модели для предсказания: {args.model}[/cyan]")
    try:
        model = YOLO(args.model)
        console.print(f"[green]Модель {args.model} успешно загружена.[/green]")
    except Exception as e:
        console.print(f"[red]Ошибка при загрузке модели {args.model}: {e}[/red]")
        return

    # Check if predict_source is a directory
    if not os.path.isdir(args.predict_source):
        console.print(f"[red]Ошибка: Папка '{args.predict_source}' не найдена.[/red]")
        return

    # Find all image files in the test folder
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.predict_source, ext)))
    
    if not image_files:
        console.print(f"[red]Ошибка: Изображения не найдены в папке '{args.predict_source}'.[/red]")
        return

    console.print(f"[cyan]Найдено {len(image_files)} изображений для обработки.[/cyan]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("[cyan]Обработка изображений...", total=len(image_files))
        for image_path in image_files:
            console.print(f"[cyan]Выполнение предсказания на '{image_path}' с conf={args.predict_conf}[/cyan]")
            try:
                results = model.predict(
                    source=image_path,
                    conf=args.predict_conf,
                    max_det=args.max_det,
                    iou=args.iou,
                    save=False
                )
            except Exception as e:
                console.print(f"[red]Ошибка во время предсказания для {image_path}: {e}[/red]")
                progress.advance(task)
                continue

            for result in results:
                img = cv2.imread(image_path)
                if img is None:
                    console.print(f"[red]Ошибка: Не удалось загрузить изображение: {image_path}[/red]")
                    continue

                grouped_objects = {}
                classes_names = result.names
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                draw = ImageDraw.Draw(pil_img)

                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    class_id = int(cls)
                    class_name = classes_names[class_id]
                    color = colors[class_id % len(colors)]

                    if class_name not in grouped_objects:
                        grouped_objects[class_name] = []
                    grouped_objects[class_name].append(box.cpu().numpy().astype(int).tolist())

                    x1, y1, x2, y2 = map(int, box)
                    draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

                    label = f"{class_name} {conf:.2f}"
                    label = label.encode("utf-8").decode("utf-8")
                    text_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
                    draw.rectangle(text_bbox, fill=(128, 128, 128))  # Gray background
                    draw.text((x1, y1 - 30), label, fill=color, font=font)

                img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                save_dir = "detected"
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, os.path.basename(image_path))
                cv2.imwrite(output_path, img_result)
                console.print(f"[green]Результат сохранен: {output_path}[/green]")

                table = Table(title="Группировка объектов", style="cyan")
                table.add_column("Класс", style="magenta")
                table.add_column("Количество", justify="right", style="green")
                for class_name, boxes in grouped_objects.items():
                    table.add_row(class_name, str(len(boxes)))
                console.print(table)

            progress.advance(task)

    console.print("[green]Предсказание завершено.[/green]")

# --- Main Menu ---
def main():
    args = parse_args()
    console.print(Panel.fit("[bold cyan]YOLOv8 Application with Rich Interface[/bold cyan]", border_style="blue"))

    while True:
        console.print("\n[bold]Выберите действие:[/bold]")
        console.print("1. Конвертировать XML в YOLO формат")
        console.print("2. Обучить модель")
        console.print("3. Выполнить предсказание на изображении")
        console.print("4. Выйти")
        choice = Prompt.ask("[bold cyan]Введите номер действия[/bold cyan]", choices=["1", "2", "3", "4"], default="4")

        if choice == "1":
            console.print(Panel.fit("[bold]Конвертация XML в YOLO формат[/bold]", border_style="green"))
            convert_xml_to_yolo_batch(args)
        elif choice == "2":
            console.print(Panel.fit("[bold]Обучение модели YOLOv8[/bold]", border_style="green"))
            train_model(args)
        elif choice == "3":
            console.print(Panel.fit("[bold]Предсказание на изображении[/bold]", border_style="green"))
            predict_image(args)
        elif choice == "4":
            console.print("[cyan]Завершение работы...[/cyan]")
            break

if __name__ == "__main__":
    main()