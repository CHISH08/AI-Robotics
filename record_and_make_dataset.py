import cv2
import os
import urllib.request
import numpy as np

# Задаем папку для хранения видео
output_folder = 'recorded_videos'
# Создаем папку, если она не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Функция для автоматического создания нового имени файла
def get_new_filename():
    # Количество уже существующих файлов в папке
    existing_files = len(os.listdir(output_folder))
    # Имя файла будет в формате video_номер.avi
    return f'{output_folder}/video_{existing_files + 1}.avi'

url = 'http://192.168.1.1:8080/?action=stream'
# Открываем видеопоток
stream = urllib.request.urlopen(url)

# Захватываем первый кадр из видеопотока
bytes = bytes()

# Переменная для хранения объекта записи
out = None
print("Начните запись видео. Нажмите Enter, чтобы остановить и сохранить видео. Повторите для записи нового видео. Нажмите 'q' для выхода.")
while True:
    # Чтение потока
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b + 2]
        bytes = bytes[b + 2:]
        # Декодируем кадр
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        # Показываем видео на экране
        cv2.imshow('Live Stream', frame)

        # Если запись включена, записываем кадр
        if out:
            out.write(frame)

        key = cv2.waitKey(1)
        # Начало или остановка записи по нажатию Enter (код клавиши 13)
        if key == 13:  # Enter
            if out:  # Если запись идет, завершаем и сохраняем файл
                print("Запись остановлена. Видео сохранено.")
                out.release()  # Останавливаем запись
                out = None
            else:  # Если запись не идет, начинаем запись нового файла
                filename = get_new_filename()
                out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))
                print(f"Начата запись в файл: {filename}")

        # Выход из программы по нажатию 'q'
        if key & 0xFF == ord('q'):
            break

# Освобождение ресурсов
if out:
    out.release()  # Если запись все еще идет, завершаем её
cv2.destroyAllWindows()
