import cv2
import numpy as np

def generate_test_video(output_path="test_video.mp4", duration_sec=10, fps=30):
    W, H = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    # Трапеция (та же, что в основном коде)
    trapezoid = np.array([
        [W*0.40, H*0.55], [W*0.60, H*0.55],
        [W*0.95, H*0.95], [W*0.05, H*0.95]
    ], dtype=np.int32)
    
    total_frames = duration_sec * fps
    
    for frame_idx in range(total_frames):
        # Рисуем "дорогу" — серый фон
        frame = np.ones((H, W, 3), dtype=np.uint8) * 120
        
        # Рисуем границы дороги (две линии)
        cv2.line(frame, (int(W*0.25), H), (int(W*0.45), int(H*0.5)), (255,255,255), 3)
        cv2.line(frame, (int(W*0.75), H), (int(W*0.55), int(H*0.5)), (255,255,255), 3)
        
        # Рисуем трапецию (отладка)
        cv2.polylines(frame, [trapezoid], True, (0,255,255), 2)
        
        # Анимируем объект: "пешеход" движется по нашей полосе
        ped_x = int(W*0.5 + 50 * np.sin(frame_idx * 0.1))
        ped_y = int(H*0.7)
        cv2.rectangle(frame, (ped_x-20, ped_y-40), (ped_x+20, ped_y), (0,0,255), -1)
        cv2.putText(frame, "PED", (ped_x-25, ped_y-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        # Анимируем объект на ПАРАЛЛЕЛЬНОЙ дороге (должен быть отсечён)
        car_x = int(W*0.15 + 30 * np.sin(frame_idx * 0.08))
        car_y = int(H*0.65)
        cv2.rectangle(frame, (car_x-30, car_y-20), (car_x+30, car_y), (255,0,0), -1)
        cv2.putText(frame, "CAR", (car_x-30, car_y-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video saved: {output_path}")

generate_test_video()