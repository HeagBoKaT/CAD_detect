import xml.etree.ElementTree as ET
import os
import glob

# --- Настройте это ---
# Словарь для сопоставления имен классов с индексами (начиная с 0)
class_mapping = {
    'Общая шероховатость': 0,
    'Технические требования': 1,
    'Линейный размер': 2,
    'Диаметральный размер': 3
    
    # Добавьте другие классы, если они появятся
}
# Пути
xml_dir = 'source_data/correct/' # Папка с вашими исходными XML
img_dir = 'source_data/correct/' # Папка с вашими исходными PNG
output_labels_dir = 'dataset/labels/train/' # Куда сохранять YOLO TXT файлы
# ---------------------

def convert_voc_to_yolo(xml_file, img_file, output_dir):
    """Конвертирует один XML файл PASCAL VOC в формат YOLO TXT."""
    base_filename = os.path.splitext(os.path.basename(xml_file))[0]
    out_txt_path = os.path.join(output_dir, base_filename + '.txt')

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            print(f"Предупреждение: Тег 'size' не найден в {xml_file}. Пропускаем.")
            # Попытка получить размер из изображения (требует Pillow или OpenCV)
            # try:
            #     from PIL import Image
            #     with Image.open(img_file) as img:
            #         img_width, img_height = img.size
            # except ImportError:
            #      print("Установите Pillow (`pip install Pillow`), чтобы получать размер из изображений.")
            #      return
            # except FileNotFoundError:
            #      print(f"Файл изображения {img_file} не найден.")
            #      return
            # Вместо этого используем значения из вашего примера XML
            # Если тега size нет, нужно либо его добавить в XML, либо получать размер из картинки
            print(f"Ошибка: Тег 'size' не найден в {xml_file}. Используются размеры из примера (1247x1760).")
            img_width = 1247
            img_height = 1760
        else:
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)

        if img_width <= 0 or img_height <= 0:
             print(f"Ошибка: Некорректные размеры изображения ({img_width}x{img_height}) в {xml_file}. Пропускаем.")
             return

        lines_to_write = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                print(f"Предупреждение: Неизвестный класс '{class_name}' в {xml_file}. Пропускаем объект.")
                continue

            class_index = class_mapping[class_name]

            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            # Конвертация в формат YOLO
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            # Нормализация
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height

            # Ограничение значений от 0 до 1 (на всякий случай)
            x_center_norm = max(0.0, min(1.0, x_center_norm))
            y_center_norm = max(0.0, min(1.0, y_center_norm))
            width_norm = max(0.0, min(1.0, width_norm))
            height_norm = max(0.0, min(1.0, height_norm))

            lines_to_write.append(f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

        # Создаем папку для вывода, если ее нет
        os.makedirs(output_dir, exist_ok=True)

        # Записываем файл, только если есть объекты для записи
        if lines_to_write:
             with open(out_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines_to_write))
             print(f"Создан файл аннотации: {out_txt_path}")
        elif os.path.exists(out_txt_path):
             # Если объектов нет, но файл существует (от прошлого запуска), удаляем его
             os.remove(out_txt_path)
             print(f"Удален пустой файл аннотации: {out_txt_path}")
        # else:
             # print(f"Нет объектов для записи в {out_txt_path}")


    except ET.ParseError:
        print(f"Ошибка: Не удалось разобрать XML файл {xml_file}.")
    except Exception as e:
        print(f"Ошибка при обработке {xml_file}: {e}")


# --- Основной цикл обработки ---
print("Начало конвертации XML в YOLO формат...")
xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
print(f"Найдено {len(xml_files)} XML файлов.")

if not xml_files:
    print(f"XML файлы не найдены в '{xml_dir}'. Проверьте путь.")
else:
    for xml_file in xml_files:
        base_filename = os.path.splitext(os.path.basename(xml_file))[0]
        img_filename = base_filename + '.png' # Предполагаем расширение .png
        img_file_path = os.path.join(img_dir, img_filename)

        if not os.path.exists(img_file_path):
             # Попробуем другие распространенные расширения
             found_img = False
             for ext in ['.jpg', '.jpeg', '.bmp']:
                 img_filename = base_filename + ext
                 img_file_path = os.path.join(img_dir, img_filename)
                 if os.path.exists(img_file_path):
                     found_img = True
                     break
             if not found_img:
                print(f"Предупреждение: Файл изображения для {xml_file} не найден в {img_dir} (проверены .png, .jpg, .jpeg, .bmp). Пропускаем.")
                continue

        # Убедимся, что папка для выходных файлов существует
        os.makedirs(output_labels_dir, exist_ok=True)
        convert_voc_to_yolo(xml_file, img_file_path, output_labels_dir)

print("Конвертация завершена.")