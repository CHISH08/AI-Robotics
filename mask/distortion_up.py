import numpy as np
import cv2

def improv_distortion(img):
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