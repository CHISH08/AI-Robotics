import cv2
import numpy as np
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)

def model_mask(predict, frame):
    # Проверяем, если есть боксы в предсказаниях
    if predict.boxes is not None:
        for box in predict.boxes:
            class_id = int(box.cls[0])  # Доступ к классу объекта
            
            if class_id in [3, 5, 9]:  # Если класс 3, 5 или 9
                # Получаем координаты бокса
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Приводим к int, чтобы использовать в индексации

                # Заполняем область внутри бокса единицами
                frame[y1:y2, x1:x2] = 1
            elif class_id in [0, 2, 7]:
                # Получаем координаты бокса
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Приводим к int, чтобы использовать в индексации

                # Заполняем область внутри бокса единицами
                frame[y1:y2, x1:x2] = 0

    return frame

def update_image(img):
    h, w = img.shape[:2]
    # Параметры камеры и искажения
    mtx = np.array([[1.17937478e+03, 0.00000000e+00, 9.24866066e+02],
                    [0.00000000e+00, 1.17865941e+03, 5.47165399e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([-0.37825373,  0.16971861, -0.00140652,  0.00480215, -0.03276766])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Коррекция искажений
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Обрезаем изображение на основе roi (регион интереса)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst

def convert_to_binary_dark(image_np, threshold=80):
    # Преобразуем изображение в градации серого
    gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Преобразуем изображение в бинарное: объекты черные, фон белый
    _, binary_image_np = cv2.threshold(gray_image_np, threshold, 255, cv2.THRESH_BINARY_INV)

    return binary_image_np

def remove_small_objects(binary_image_np, min_area):
    # Находим все контуры на бинарной маске
    contours, _ = cv2.findContours(binary_image_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Проходим по всем найденным контурам
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Если площадь контура меньше минимальной, заполняем этот контур черным цветом
        if area < min_area:
            cv2.drawContours(binary_image_np, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    
    return binary_image_np


def extract_evenly_spaced_frames(video_path, num_frames=10):
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка открытия видеофайла")
        return

    # Получаем общее количество кадров в видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Рассчитываем шаг между кадрами для равномерного извлечения
    step = total_frames // num_frames

    # Извлекаем 10 равномерно распределенных кадров
    for i in range(num_frames):
        frame_number = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Переход к нужному кадру

        ret, frame = cap.read()
        if ret:
            # Сохраняем каждый кадр как np.array
            frame = update_image(frame)
            binary_frame = convert_to_binary_dark(frame)
            binary_frame_cleaned = remove_small_objects(binary_frame, min_area=100)

            predict = model.predict(frame, conf=0.8)[0].cpu()
            fin_frame = model_mask(predict, binary_frame_cleaned)

            # Сохранение кадра как np.array в формате .npy
            frame_filename = f"./output_frames/frame_{i}.npy"
            np.save(frame_filename, binary_frame)
            print(f"Сохранён кадр {i + 1} как {frame_filename}")
        else:
            print(f"Не удалось прочитать кадр {frame_number}")

    # Закрываем видеофайл
    cap.release()


model = YOLO("./AI-Robotics/top_with_down/vers2/best.pt")
video_file = "./AI-Robotics/image_and_data/final2.avi"  # Замените на путь к вашему видеофайлу
extract_evenly_spaced_frames(video_file, num_frames=10)
