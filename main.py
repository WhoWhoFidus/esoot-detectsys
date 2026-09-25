import cv2
import numpy as np

# Параметры кадра (стандарт для Raspberry Pi Camera / веб-камеры)
WIDTH, HEIGHT = 640, 480

# Координаты трапеции (подбираются экспериментально под положение камеры)
# Точки идут по часовой стрелке: левый-верх, правый-верх, правый-низ, левый-низ
TRAPEZOID = np.array([
    [WIDTH * 0.40, HEIGHT * 0.55],  # левый верх (дальняя зона)
    [WIDTH * 0.60, HEIGHT * 0.55],  # правый верх
    [WIDTH * 0.95, HEIGHT * 0.95],  # правый низ (ближняя зона)
    [WIDTH * 0.05, HEIGHT * 0.95],  # левый низ
], dtype=np.int32)


def apply_roi_mask(frame, trapezoid):
    """Оставляет только область трапеции, остальное — чёрное."""
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [trapezoid], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)

def preprocess(frame):
    """Серый → размытие → Canny."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Гауссово размытие убирает шум, но сохраняет границы
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Пороги Canny подбираются под освещение (можно сделать адаптивными)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_lanes(edges):
    """Возвращает список кортежей (x1, y1, x2, y2)."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=20
    )
    if lines is None:
        return []
    return [tuple(int(v) for v in line.ravel()) for line in lines]


def split_lanes(lines, width):
    """Делит кортежи линий на левые и правые."""
    left_lines, right_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line  # уже кортеж из 4 int

        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) < 0.3:
            continue

        if slope < 0 and x1 < width // 2:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > width // 2:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def average_line(lines, height):
    """Усредняет набор линий в одну."""
    if not lines:
        return None

    points = []
    for x1, y1, x2, y2 in lines:
        points.append([x1, y1])
        points.append([x2, y2])
    points = np.array(points)

    if len(np.unique(points[:, 1])) < 2:
        return None

    fit = np.polyfit(points[:, 1], points[:, 0], 1)  # x = a*y + b
    y1, y2 = height, int(height * 0.6)
    x1, x2 = int(np.polyval(fit, y1)), int(np.polyval(fit, y2))
    return (x1, y1, x2, y2)

def is_on_path(bbox, left_lane, right_lane, width):
    """
    Проверяет, находится ли объект между границами полосы.
    bbox: (x, y, w, h) от YOLO
    """
    x_center = bbox[0] + bbox[2] // 2
    y_bottom = bbox[1] + bbox[3]  # нижняя точка объекта (контакт с землёй)

    # Если границы не найдены — считаем, что весь кадр наш (fallback)
    if left_lane is None or right_lane is None:
        return True

    # Интерполируем x левой и правой границы на высоте y_bottom
    left_x = interpolate_x(left_lane, y_bottom)
    right_x = interpolate_x(right_lane, y_bottom)

    return left_x < x_center < right_x


def interpolate_x(line, y):
    """Возвращает x точки на линии при заданном y."""
    x1, y1, x2, y2 = line
    if y2 == y1:
        return x1
    t = (y - y1) / (y2 - y1)
    return int(x1 + t * (x2 - x1))

def process_frame(frame, yolo_results=None):
    """Основной пайплайн обработки одного кадра."""
    # 1. ROI
    roi = apply_roi_mask(frame, TRAPEZOID)

    # 2. Canny
    edges = preprocess(roi)

    cv2.imshow("Canny edges", edges)

    # 2. Сохранить ROI для просмотра
    cv2.imshow("ROI mask", roi)

    # 3. Hough
    lines = detect_lanes(edges)

    # 4. Разделение
    left, right = split_lanes(lines, WIDTH)
    left_lane = average_line(left, HEIGHT)
    right_lane = average_line(right, HEIGHT)

    # 5. Проверка объектов (если есть детекции от YOLO)
    on_path_objects = []
    if yolo_results is not None:
        for bbox, class_id in yolo_results:
            if is_on_path(bbox, left_lane, right_lane, WIDTH):
                on_path_objects.append((bbox, class_id))

    # 6. Визуализация для отладки
    debug = frame.copy()
    cv2.polylines(debug, [TRAPEZOID], True, (0, 255, 255), 2)
    if left_lane:
        cv2.line(debug, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 0, 255), 3)
    if right_lane:
        cv2.line(debug, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 0, 255), 3)

    return debug, on_path_objects

cap = cv2.VideoCapture(0)  # или путь к тестовому видео

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    debug, objects = process_frame(frame)

    cv2.imshow("Detection Zone + Canny", debug)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#R77041