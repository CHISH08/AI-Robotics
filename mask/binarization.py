import cv2

def convert_to_binary_dark(image_np, threshold=90):
    # Проверяем, если изображение не является одноканальным, преобразуем его в градации серого
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Преобразуем изображение в бинарное: объекты черные, фон белый
    _, binary_image_np = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY_INV)

    binary_image_np = (remove_small_objects(binary_image_np, 100) / 255).astype(int)

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
