"""
ВКР: Веб-ориентированная интеллектуальная система компьютерного зрения
для обеспечения безопасности средств индивидуальной мобильности (СИМ)

Модуль бортового анализа видеопотока:
    ROI (трапеция) → Canny → Hough → YOLO → фильтр "на пути"
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================

WIDTH, HEIGHT = 640, 480

# Трапеция — зона перед самокатом с учётом перспективы
# Точки по часовой стрелке: левый-верх, правый-верх, правый-низ, левый-низ
TRAPEZOID = np.array([
    [int(WIDTH * 0.40), int(HEIGHT * 0.55)],
    [int(WIDTH * 0.60), int(HEIGHT * 0.55)],
    [int(WIDTH * 0.95), int(HEIGHT * 0.95)],
    [int(WIDTH * 0.05), int(HEIGHT * 0.95)],
], dtype=np.int32)

# Классы COCO, которые нас интересуют (для отладки)
# 0 — person, 2 — car, 9 — traffic light, 11 — stop sign
TARGET_CLASSES = {0, 2, 9, 11}

CLASS_NAMES = {
    0: "person",
    2: "car",
    9: "traffic light",
    11: "stop sign",
}

CLASS_COLORS = {
    0: (0, 0, 255),      # пешеход — красный
    2: (255, 0, 0),      # машина — синий
    9: (0, 255, 255),    # светофор — жёлтый
    11: (255, 0, 255),   # знак — пурпурный
}


# ============================================================
# 2. ROI — МАСКА ТРАПЕЦИИ
# ============================================================

def apply_roi_mask(frame, trapezoid):
    """Оставляет только область трапеции, остальное — чёрное."""
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [trapezoid], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


# ============================================================
# 3. ПРЕДОБРАБОТКА — CANNY
# ============================================================

def preprocess(frame):
    """Серый → размытие → Canny."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    return edges


# ============================================================
# 4. ПОИСК ЛИНИЙ — HOUGH
# ============================================================

def detect_lanes(edges):
    """Возвращает список кортежей (x1, y1, x2, y2)."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=70,
        maxLineGap=20
    )
    if lines is None:
        return []
    return [tuple(int(v) for v in line.ravel()) for line in lines]


# ============================================================
# 5. РАЗДЕЛЕНИЕ ЛИНИЙ НА ЛЕВУЮ И ПРАВУЮ
# ============================================================

def split_lanes(lines, width):
    """Делит линии на левые и правые по знаку углового коэффициента."""
    left_lines, right_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line

        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Отсекаем горизонтальные линии (разметка, тени)
        if abs(slope) < 0.3:
            continue

        if slope < 0 and x1 < width // 2:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and x1 > width // 2:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def average_line(lines, height):
    """Усредняет набор линий в одну (снижение дрожания)."""
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


# ============================================================
# 6. ПРОВЕРКА "ОБЪЕКТ НА ПУТИ"
# ============================================================

def interpolate_x(line, y):
    """Возвращает x точки на линии при заданном y."""
    x1, y1, x2, y2 = line
    if y2 == y1:
        return x1
    t = (y - y1) / (y2 - y1)
    return int(x1 + t * (x2 - x1))


def is_on_path(bbox, left_lane, right_lane, width):
    """
    Проверяет, находится ли объект между границами полосы.
    bbox: (x, y, w, h)
    """
    x_center = bbox[0] + bbox[2] // 2
    y_bottom = bbox[1] + bbox[3]

    if left_lane is None or right_lane is None:
        return True  # fallback

    left_x = interpolate_x(left_lane, y_bottom)
    right_x = interpolate_x(right_lane, y_bottom)

    return left_x < x_center < right_x


def is_inside_trapezoid(bbox, trapezoid):
    """Проверяет, попадает ли нижняя точка объекта в трапецию."""
    x_center = bbox[0] + bbox[2] // 2
    y_bottom = bbox[1] + bbox[3]
    return cv2.pointPolygonTest(trapezoid, (x_center, y_bottom), False) >= 0


# ============================================================
# 7. ДЕТЕКЦИЯ ОБЪЕКТОВ — YOLO
# ============================================================

def detect_objects(model, frame, conf=0.4):
    """
    Возвращает список bbox в формате (x, y, w, h, class_id, confidence).
    """
    results = model.predict(
        source=frame,
        conf=conf,
        verbose=False,
        device="cpu"
    )

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            conf_val = float(box.conf[0])

            detections.append((x, y, w, h, cls_id, conf_val))

    return detections


# ============================================================
# 8. ОСНОВНОЙ ПАЙПЛАЙН ОБРАБОТКИ КАДРА
# ============================================================

def process_frame(frame, model):
    """ROI → Canny → Hough → YOLO → фильтр 'на пути'."""

    # 1. ROI
    roi = apply_roi_mask(frame, TRAPEZOID)

    # 2. Canny
    edges = preprocess(roi)

    # 3. Hough + границы полосы
    lines = detect_lanes(edges)
    left, right = split_lanes(lines, WIDTH)
    left_lane = average_line(left, HEIGHT)
    right_lane = average_line(right, HEIGHT)

    # 4. YOLO
    detections = detect_objects(model, frame)

    # 5. Фильтр "на пути"
    on_path_objects = []
    for (x, y, w, h, cls_id, conf) in detections:
        bbox = (x, y, w, h)

        # Отсечка по трапеции
        if not is_inside_trapezoid(bbox, TRAPEZOID):
            continue

        # Уточнение по линиям полосы
        if left_lane and right_lane:
            on_path = is_on_path(bbox, left_lane, right_lane, WIDTH)
        else:
            on_path = True

        if on_path:
            on_path_objects.append((bbox, cls_id, conf))

    # 6. Визуализация
    debug = frame.copy()

    # Трапеция
    cv2.polylines(debug, [TRAPEZOID], True, (0, 255, 255), 2)

    # Границы полосы
    if left_lane:
        cv2.line(debug, (left_lane[0], left_lane[1]),
                 (left_lane[2], left_lane[3]), (0, 0, 255), 3)
    if right_lane:
        cv2.line(debug, (right_lane[0], right_lane[1]),
                 (right_lane[2], right_lane[3]), (0, 0, 255), 3)

    # Объекты
    on_path_keys = {(tuple(o[0]), o[1]) for o in on_path_objects}

    for (x, y, w, h, cls_id, conf) in detections:
        bbox = (x, y, w, h)
        is_on = (tuple(bbox), cls_id) in on_path_keys

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        thickness = 3 if is_on else 1

        cv2.rectangle(debug, (x, y), (x + w, y + h), color, thickness)

        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
        if is_on:
            label += " | ON PATH"

        cv2.putText(debug, label, (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return debug, on_path_objects, edges, roi


# ============================================================
# 9. ТОЧКА ВХОДА
# ============================================================

def main():
    print("[INFO] Загрузка YOLO...")
    model = YOLO("yolov8n.pt")  # скачается автоматически при первом запуске

    # Источник: 0 — веб-камера; либо путь к видеофайлу
    source = 0
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Не удалось открыть источник видео")
        return

    print("[INFO] Запуск. Нажмите 'q' для выхода, 'e' — показать edges/ROI, 'r' — скрыть.")

    show_debug = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Конец видео")
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        debug, on_path, edges, roi = process_frame(frame, model)

        cv2.imshow("YOLO + ROI + Lanes", debug)

        if show_debug:
            cv2.imshow("Canny edges", edges)
            cv2.imshow("ROI mask", roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            show_debug = True
        elif key == ord('r'):
            show_debug = False
            cv2.destroyWindow("Canny edges")
            cv2.destroyWindow("ROI mask")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()