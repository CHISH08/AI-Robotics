from ultralytics import YOLO
import logging
import torch

logging.getLogger("ultralytics").setLevel(logging.ERROR)

class ModelYOLO(YOLO):
    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """
        Инициализация модели YOLO.
        :param model: Имя модели, которую мы будем использовать.
        :param task: Задача, для которой загружается модель.
        :param verbose: Флаг для вывода логов.
        """
        super().__init__(model, task, verbose)

    def get_predictions(self, frame):
        """
        Выполняет предсказания на одном кадре.
        :param frame: Изображение, на котором будет происходить предсказание.
        :return: Результаты предсказания (тензоры с боксами, классами и уверенностями).
        """
        predictions = self.predict(source=frame, imgsz=640, conf=0.1)
        result = predictions[0]  # Берем первый результат, если обрабатываем один кадр
        return result.boxes.xyxy, result.boxes.conf, result.boxes.cls

    def create_mask(self, classes, confs, class_id, threshold):
        """
        Создает маску для фильтрации боксов по классу и порогу уверенности.
        :param classes: Тензор с классами.
        :param confs: Тензор с уверенностями.
        :param class_id: Интересующий класс.
        :param threshold: Порог уверенности.
        :return: Булевый тензор, который можно использовать для маскировки данных.
        """
        return (classes == class_id) & (confs >= threshold)

    def filter_predictions(self, boxes, confs, classes, class_conf_thresholds, class_limits):
        """
        Фильтрует предсказания по порогам уверенности для каждого класса и ограничивает количество объектов.
        :param boxes: Тензор с координатами боксов.
        :param confs: Тензор с уверенностями для каждого бокса.
        :param classes: Тензор с классами для каждого бокса.
        :param class_conf_thresholds: Список порогов уверенности для каждого класса.
        :param class_limits: Список лимитов (максимальное количество объектов) для каждого класса.
        :return: Отфильтрованные боксы с учетом порогов уверенности и лимитов для каждого класса.
        """
        filtered_boxes = {}

        # Проходим по каждому классу и его порогу уверенности
        for class_id, conf_threshold in enumerate(class_conf_thresholds):
            # Создаем маску для текущего класса и его порога уверенности
            mask = self.create_mask(classes, confs, class_id, conf_threshold)

            # Проверяем, есть ли хотя бы один box, который прошел фильтр
            if torch.any(mask):
                class_boxes = boxes[mask]
                class_confs = confs[mask]

                # Сортируем боксы по уверенности (по убыванию)
                sorted_indices = torch.argsort(class_confs, descending=True)
                sorted_boxes = class_boxes[sorted_indices]
                sorted_confs = class_confs[sorted_indices]

                # Применяем лимит на количество объектов для текущего класса
                limit = class_limits[class_id]
                limited_boxes = sorted_boxes[:limit]
                limited_confs = sorted_confs[:limit]

                # Добавляем боксы для текущего класса
                filtered_boxes[int(class_id)] = limited_boxes.cpu().numpy()

        return filtered_boxes

    def get_all_classes_coordinates(self, frame, class_conf_thresholds, class_limits):
        """
        Метод для получения координат боксов для всех классов с учетом заданных порогов уверенности и лимитов для каждого класса.
        :param frame: Изображение, на котором выполняется предсказание.
        :param class_conf_thresholds: Список порогов уверенности для каждого класса.
        :param class_limits: Список лимитов (максимальное количество объектов) для каждого класса.
        :return: Словарь с координатами и уверенностями для всех классов, удовлетворяющих порогу уверенности и лимитам.
        """
        boxes, confs, classes = self.get_predictions(frame)
        return self.filter_predictions(boxes, confs, classes, class_conf_thresholds, class_limits)

    def get_interest_coordinates(self, frame, id_cls, confs):
        """
        Главный метод для получения интересующих координат боксов для определенных классов.
        :param frame: Изображение, на котором выполняется предсказание.
        :param id_cls: Список классов, которые нас интересуют (идентификаторы классов).
        :param confs: Список порогов уверенности для интересующих классов.
        :return: Координаты для интересующих классов.
        """
        # Получаем предсказания
        boxes, pred_confs, pred_classes = self.get_predictions(frame)

        # Отфильтрованные боксы
        filtered_boxes = {}

        # Проходим по интересующим классам и их порогам уверенности
        for class_id, conf_threshold in zip(id_cls, confs):
            # Создаем маску для текущего интересующего класса и его порога уверенности
            mask = self.create_mask(pred_classes, pred_confs, class_id, conf_threshold)

            # Проверяем, есть ли хотя бы один box, который прошел фильтр
            if torch.any(mask):
                class_boxes = boxes[mask]
                class_confs = pred_confs[mask]

                # Сортируем боксы по уверенности (по убыванию)
                sorted_indices = torch.argsort(class_confs, descending=True)
                sorted_boxes = class_boxes[sorted_indices]
                sorted_confs = class_confs[sorted_indices]

                # Добавляем боксы для текущего класса
                filtered_boxes[int(class_id)] = sorted_boxes.cpu().numpy()[0]

        return filtered_boxes
