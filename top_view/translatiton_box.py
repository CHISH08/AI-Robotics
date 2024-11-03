from mask import improv_distortion, convert_to_binary_dark, ModelYOLO
import cv2
import numpy as np
import pickle
import time

def find_cube(image, kernel_size=(150, 150), threshold=0.95):
    """
    Ищет на изображении область, где процент вхождений 1 в заданном окне равен 95%.
    
    :param image: Входное изображение (numpy array), состоящее из 0 и 1.
    :param kernel_size: Размер ядра (окна), в котором ищем процент вхождения.
    :param threshold: Порог вхождения (процент единиц) для поиска.
    :return: Координаты (x, y) левого верхнего угла области, если найдена, иначе None.
    """
    h, w = image.shape
    kernel_h, kernel_w = kernel_size

    # Проход по изображению с окном (ядром) заданного размера
    for y in range(h - kernel_h + 1):
        for x in range(w - kernel_w + 1):
            # Извлечение окна (ядра)
            window = image[y:y+kernel_h, x:x+kernel_w]
            
            # Подсчёт количества единиц в окне
            ones_count = np.sum(window == 1)
            total_count = kernel_h * kernel_w
            
            # Подсчёт процента единиц в окне
            percentage = ones_count / total_count
            
            # Если процент единиц больше или равен порогу, возвращаем координаты
            if percentage >= threshold:
                return x, y

    # Если не найдено ни одного окна, которое удовлетворяет условию
    return None

def update_mask(image):
    # Преобразуем изображение в формат uint8, если оно не в этом формате
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Найдем контуры, чтобы определить границы квадрата
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Найдем самый большой контур (предположительно квадрат)
    largest_contour = max(contours, key=cv2.contourArea)

    # Создадим маску с нулями, того же размера что и изображение
    mask = np.zeros_like(image)
    
    # Заполним маску контуром квадрата
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Применим расширение только внутри маски
    kernel = np.ones((7, 7), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    
    # Ограничиваем действие морфологической операции только в пределах маски
    result = np.where(mask == 255, dilated_image, image)
    
    return result

class DisplayVideo:
    def __init__(self, model_path='/home/denis/code/test_yolo/top_with_down/vers2/best.pt', url="rtsp://Admin:rtf123@192.168.2.250/251:554/1/1"):
        ip_camera_url_left = url
        self.cap = cv2.VideoCapture(ip_camera_url_left)
        if not self.cap.isOpened():
            print("Ошибка открытия видеофайла")
            self.cap.release()
            cv2.destroyAllWindows()
            exit()
        self.model = ModelYOLO(model_path)
        # self.class_names = ['Ball', 'Base green', 'Base red', 'Basket', 'Button blue', 'Button box with red', 'Button green', 'Button red', 'Cube', 'Enemy']
        self.class_names = list(self.model.names.values())
        self.conf_class_for_top = [0.1,0.4,0.4,0.4,0.3,0.3,0.2,0.6,0.3,0.5]
        self.conf_class_for_down = [0.75,0.65,0.65,0.6,0.6,0.6,0.6,0.6,0.7,0.7]
        self.class_limits = [1,1,1,2,2,3,2,3,2,4]
        # self.class_limits = [1,1,1,2,2,2,2,2,2,1]
        self.priority = [1, 2, 5, 3, 4, 6, 7, 9, 8, 0]
        self.class_colors = [
            [0, 0, 0],
            [255, 255, 255],
            [255, 165, 0],    # оранжевый
            [0, 128, 0],      # зеленый
            [255, 0, 0],      # красный
            [128, 128, 128],  # серый (Basket)
            [135, 206, 235],  # голубой (Button blue)
            [127, 255, 0],    # салатовый
            [245, 245, 220],  # бежевый
            [255, 255, 0],    # желтый
            [128, 0, 0],      # бордовый
            [0, 0, 255]       # синий
        ]

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def update_cadr_for_top(self, frame):
        updated_frame = improv_distortion(frame)
        combined_frame = convert_to_binary_dark(updated_frame)
        combined_frame = update_mask(combined_frame)

        # Используем модель YOLO для предсказаний
        classes_coords = self.model.get_all_classes_coordinates(updated_frame, self.conf_class_for_top, self.class_limits)

        for id_cls, xyxys in classes_coords.items():
            for x1, y1, x2, y2 in xyxys:
                x1 = np.floor(x1).astype(int)  # Округляем влево вниз
                y1 = np.floor(y1).astype(int)  # Округляем вверх вниз
                x2 = np.ceil(x2).astype(int)   # Округляем вправо вверх
                y2 = np.ceil(y2).astype(int)   # Округляем вниз вверх
                combined_frame[y1:y2, x1:x2] = id_cls + 2

        return combined_frame

    def update_cadr_for_down(self, frame):
        updated_frame = improv_distortion(frame)
        combined_frame = convert_to_binary_dark(updated_frame)

        # Используем модель YOLO для предсказаний
        classes_coords = self.model.get_all_classes_coordinates(updated_frame, self.conf_class_for_down, self.class_limits)
        # print(classes_coords)
        classes_coords = dict(sorted(classes_coords.items(), key=lambda x: self.priority.index(x[0])))
        # print(sorted_classes_coords)

        for id_cls, xyxys in classes_coords.items():
            xyxys = xyxys["boxes"]
            for x1, y1, x2, y2 in xyxys:
                x1 = np.floor(x1).astype(int)  # Округляем влево вниз
                y1 = np.floor(y1).astype(int)  # Округляем вверх вниз
                x2 = np.ceil(x2).astype(int)   # Округляем вправо вверх
                y2 = np.ceil(y2).astype(int)   # Округляем вниз вверх
                combined_frame[y1:y2, x1:x2] = id_cls + 2

        return combined_frame

    def update_cadr_for_top_save(self, frame):
        updated_frame = improv_distortion(frame)
        combined_frame = convert_to_binary_dark(updated_frame)
        combined_frame = update_mask(combined_frame)

        # Используем модель YOLO для предсказаний
        classes_coords = self.model.get_all_classes_coordinates(updated_frame, self.conf_class_for_top, self.class_limits)
        classes_coords_new = {}

        for id_cls, xyxys in classes_coords.items():
            if id_cls in {0, 1, 2, 5, 8, 9}:
                for x1, y1, x2, y2 in xyxys:
                    x1 = np.floor(x1).astype(int)  # Округляем влево вниз
                    y1 = np.floor(y1).astype(int)  # Округляем вверх вниз
                    x2 = np.ceil(x2).astype(int)   # Округляем вправо вверх
                    y2 = np.ceil(y2).astype(int)   # Округляем вниз вверх
                    combined_frame[y1:y2, x1:x2] = 0
            if id_cls in {3, 5, 8, 9}:
                for x1, y1, x2, y2 in xyxys:
                    x1 = np.floor(x1).astype(int)  # Округляем влево вниз
                    y1 = np.floor(y1).astype(int)  # Округляем вверх вниз
                    x2 = np.ceil(x2).astype(int)   # Округляем вправо вверх
                    y2 = np.ceil(y2).astype(int)   # Округляем вниз вверх
                    combined_frame[y1:y2, x1:x2] = id_cls+2

        return combined_frame, classes_coords_new

    def save_cadr(self):
        for i in range(1):
            frame = self.get_frame()
            if frame is None:
                print("Конец видео")
                break
            time_first = time.time()
            update_frame, classes = self.update_cadr_for_top_save(frame)
            time_fin = time.time()
            np.save(f"numpy_data/frame_{i}.npy", update_frame)
            with open(f"numpy_data/boxes_{i}.pickle", 'wb') as handle:
                pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.disconnect()

    def get_static_cadr(self, output_path, frame_rate=30):
        screen_width = 1768  # Замените на ширину вашего экрана
        screen_height = 912  # Замените на высоту вашего экрана
        
        # Определяем кодек и создаем объект VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (screen_width, screen_height))

        for i in range(1):
            frame = self.get_frame()
            if frame is None:
                print("Конец видео")
                break

            update_frame, classes = self.update_cadr_for_top_save(frame)
            out.write(update_frame)

            # Нажмите 'q', чтобы выйти из цикла
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Освобождаем ресурсы
        out.release()
        self.disconnect()


    def broadcast_visualization(self):
        # Получаем разрешение экрана
        screen_width = 1768  # Замените на ширину вашего экрана
        screen_height = 912  # Замените на высоту вашего экрана

        while self.cap.isOpened():
            frame = self.get_frame()
            if frame is None:
                print("Конец видео")
                break

            update_frame = self.update_cadr_for_top(frame)

            # Создаем пустой трехмерный тензор для хранения цветов (RGB) с такими же размерами, как update_frame
            colored_frame = np.zeros((*update_frame.shape, 3), dtype=np.uint8)

            # Наложение цветов на основе классов
            for class_id in range(len(self.class_colors)):
                mask = update_frame == class_id  # Маска для текущего класса
                colored_frame[mask] = self.class_colors[class_id]

            # Преобразуем из RGB в BGR для отображения
            colored_frame_bgr = cv2.cvtColor(colored_frame, cv2.COLOR_RGB2BGR)

            # Объединяем маскированное изображение и реальный кадр
            colored_frame_bgr = cv2.resize(colored_frame_bgr, (screen_width//2, screen_height))  # Уменьшаем изображение до размеров окна
            real_frame = improv_distortion(frame)
            predict = self.model.predict(real_frame)
            real_frame = predict[0].plot()
            real_frame = cv2.resize(real_frame, (screen_width//2, screen_height))

            combined_frame = np.hstack((colored_frame_bgr, real_frame))

            # Отображаем объединенный кадр
            cv2.imshow("Masked (Left) and Real (Right)", combined_frame)

            # Нажмите 'q', чтобы выйти из цикла
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.disconnect()

    def save_partial_video(self, output_path, rate=0.01, frame_rate=30):
        """
        Метод для сохранения первых n * rate кадров видео.
        
        :param output_path: Путь для сохранения выходного видео.
        :param rate: Процент кадров для сохранения (от 0 до 1).
        :param frame_rate: Частота кадров (fps) для выходного видео.
        """
        # Получаем разрешение экрана
        screen_width = 1768  # Замените на ширину вашего экрана
        screen_height = 912  # Замените на высоту вашего экрана
        
        # Определяем кодек и создаем объект VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (screen_width, screen_height))

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Общее количество кадров
        frames_to_save = int(total_frames * rate)  # Количество кадров для сохранения

        for _ in range(frames_to_save):
            if not self.cap.isOpened():
                print("Не удалось открыть видео.")
                break

            frame = self.get_frame()
            if frame is None:
                print("Конец видео")
                break

            # Обновляем кадр (если нужно)
            update_frame, _ = self.update_cadr_for_top_save(frame)

            # Создаем пустой трехмерный тензор для хранения цветов (RGB)
            colored_frame = np.zeros((*update_frame.shape, 3), dtype=np.uint8)

            # Наложение цветов на основе классов
            for class_id in range(len(self.class_colors)):
                mask = update_frame == class_id  # Маска для текущего класса
                colored_frame[mask] = self.class_colors[class_id]

            # Преобразуем из RGB в BGR для отображения
            colored_frame_bgr = cv2.cvtColor(colored_frame, cv2.COLOR_RGB2BGR)

            # Применяем модель к реальному кадру
            real_frame = improv_distortion(frame)
            predict = self.model.predict(real_frame)
            real_frame = predict[0].plot()
            colored_frame_bgr = cv2.resize(colored_frame_bgr, (screen_width//2, screen_height))  # Уменьшаем изображение до размеров
            real_frame = cv2.resize(real_frame, (screen_width//2, screen_height))

            # Объединяем кадры (если нужно)
            combined_frame = np.hstack((colored_frame_bgr, real_frame))

            # Сохраняем объединенный кадр
            out.write(combined_frame)

            # Нажмите 'q', чтобы выйти из цикла
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Освобождаем ресурсы
        out.release()
        self.disconnect()

    def disconnect(self):
        self.cap.release()
        cv2.destroyAllWindows()
