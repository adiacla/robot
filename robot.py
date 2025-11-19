import cv2
import math
import numpy as np
import mediapipe as mp
import pygame
from collections import deque
import time

# ---------- Config MediaPipe ----------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ---------- Pygame robot drawing helpers ----------
class RobotVisualizer:
    def __init__(self, width=640, height=480):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Robot mimic")
        self.clock = pygame.time.Clock()

    def to_screen(self, x, y):
        # x,y en pixeles ya están listos, solo invertimos Y si hace falta
        return int(x), int(y)

    def draw_robot(self, shoulder_center, left_shoulder, right_shoulder, left_angle, right_angle, arm_length=80):
        self.screen.fill((30, 30, 30))

        # Torso
        cx, cy = self.width // 2, self.height // 2
        pygame.draw.rect(self.screen, (200, 200, 200), (cx-20, cy-40, 40, 80))

        # Shoulders positions (if none provided, usar valores por defecto)
        if shoulder_center is None:
            shoulder_center = (cx, cy-40)
        if left_shoulder is None:
            left_shoulder = (cx-20, cy-40)
        if right_shoulder is None:
            right_shoulder = (cx+20, cy-40)

        # draw shoulders
        pygame.draw.circle(self.screen, (255, 100, 100), shoulder_center, 6)
        pygame.draw.circle(self.screen, (100, 255, 100), left_shoulder, 5)
        pygame.draw.circle(self.screen, (100, 100, 255), right_shoulder, 5)

        # Arms: angle in degrees, 0 = horizontal to the right
        # Left arm
        lx, ly = left_shoulder
        la = math.radians(left_angle)
        left_elbow = (int(lx + arm_length * math.cos(la)),
                      int(ly - arm_length * math.sin(la)))
        pygame.draw.line(self.screen, (200, 200, 0), (lx, ly), left_elbow, 8)
        pygame.draw.circle(self.screen, (255, 255, 255), left_elbow, 6)

        # Right arm
        rx, ry = right_shoulder
        ra = math.radians(right_angle)
        right_elbow = (int(rx + arm_length * math.cos(ra)),
                       int(ry - arm_length * math.sin(ra)))
        pygame.draw.line(self.screen, (200, 200, 0), (rx, ry), right_elbow, 8)
        pygame.draw.circle(self.screen, (255, 255, 255), right_elbow, 6)

        pygame.display.flip()
        self.clock.tick(30)

# ---------- Utility functions ----------
def angle_between(a, b, c):
    # devuelve ángulo en grados en 'b' formado por ab y cb
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return ang

def smooth(value, prev, alpha=0.6):
    if prev is None:
        return value
    return alpha * value + (1-alpha) * prev

# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(0)
    # ajustar el tamaño para performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vis = RobotVisualizer(640, 480)

    prev_left_angle = None
    prev_right_angle = None

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        last_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # manejo de eventos pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    return

            # Flip en X para efecto espejo (opcional)
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image_rgb)

            # Obtener landmarks si existen
            h, w, _ = frame.shape
            left_wrist = None
            right_wrist = None
            left_elbow = None
            right_elbow = None
            left_shoulder = None
            right_shoulder = None
            shoulder_center = None

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # indices de MediaPipe Pose
                L_SHOULDER = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
                R_SHOULDER = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
                L_ELBOW = mp_holistic.PoseLandmark.LEFT_ELBOW.value
                R_ELBOW = mp_holistic.PoseLandmark.RIGHT_ELBOW.value
                L_WRIST = mp_holistic.PoseLandmark.LEFT_WRIST.value
                R_WRIST = mp_holistic.PoseLandmark.RIGHT_WRIST.value
                L_HIP = mp_holistic.PoseLandmark.LEFT_HIP.value
                R_HIP = mp_holistic.PoseLandmark.RIGHT_HIP.value

                def to_px(lm_item):
                    return int(lm_item.x * w), int(lm_item.y * h)

                left_shoulder = to_px(lm[L_SHOULDER])
                right_shoulder = to_px(lm[R_SHOULDER])
                left_elbow = to_px(lm[L_ELBOW])
                right_elbow = to_px(lm[R_ELBOW])
                left_wrist = to_px(lm[L_WRIST])
                right_wrist = to_px(lm[R_WRIST])

                shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                                   (left_shoulder[1] + right_shoulder[1]) // 2)

                # computar ángulos de codo (en el codo, entre hombro-elbow-muñeca)
                left_elbow_angle = angle_between(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = angle_between(right_shoulder, right_elbow, right_wrist)

                # suavizado
                left_elbow_angle = smooth(left_elbow_angle, prev_left_angle, alpha=0.5)
                right_elbow_angle = smooth(right_elbow_angle, prev_right_angle, alpha=0.5)
                prev_left_angle = left_elbow_angle
                prev_right_angle = right_elbow_angle

                # Detectar estados:
                # - brazo extendido (abierto) si el ángulo del codo > 160 (casi recto)
                # - brazo cerrado si el ángulo < 100 (flexionado)
                left_extended = left_elbow_angle > 160
                right_extended = right_elbow_angle > 160
                arms_open = left_extended and right_extended
                arms_closed = (left_elbow_angle < 100) and (right_elbow_angle < 100)

                # manos arriba/abajo: comparar y de la muñeca contra hombro o cadera
                # recordar: en pixel coords, y pequeño = arriba
                LEFT_HAND_UP = None
                RIGHT_HAND_UP = None
                if left_wrist and left_shoulder:
                    LEFT_HAND_UP = left_wrist[1] < left_shoulder[1]  # True si muñeca arriba del hombro
                if right_wrist and right_shoulder:
                    RIGHT_HAND_UP = right_wrist[1] < right_shoulder[1]

                # Opcional: imprimir / mostrar en pantalla
                status_text = f"Left angle: {int(left_elbow_angle)} Right angle: {int(right_elbow_angle)}"
                status_text += f" | Arms open: {arms_open} closed: {arms_closed}"
                status_text += f" | L_up: {LEFT_HAND_UP} R_up: {RIGHT_HAND_UP}"

                # dibujar landmarks en frame para debug
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Mapeo a ángulos para robot (simplificado): usar ángulo del hombro horizontalizado
                # Para visualizar en pygame convertimos ángulo de codo a ángulo del hombro simple
                # Aquí lo hacemos muy simple: si el brazo está extendido, dibujamos horizontal (0 o 180)
                # si está levantado, aumentamos el ángulo.
                # Más avanzado: calcular ángulo del hombro (vector hombro->muñeca)
                def shoulder_arm_angle(shoulder, wrist):
                    vx = wrist[0] - shoulder[0]
                    vy = shoulder[1] - wrist[1]  # invertimos Y para tener eje usual
                    ang = math.degrees(math.atan2(vy, vx))  # 0 = right, 90 = up
                    return ang

                left_shoulder_ang = shoulder_arm_angle(left_shoulder, left_wrist) if left_wrist else 180
                right_shoulder_ang = shoulder_arm_angle(right_shoulder, right_wrist) if right_wrist else 0

                # Ajustes para dibujo (queremos ángulos en 0..360 y ciertos offsets)
                left_draw_angle = left_shoulder_ang
                right_draw_angle = right_shoulder_ang

                # Mostrar info sobre el frame
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Render robot con ángulos
                vis.draw_robot(shoulder_center, left_shoulder, right_shoulder, left_draw_angle, right_draw_angle)
            else:
                # si no detecta, dibujar robot por defecto
                vis.draw_robot((vis.width//2, vis.height//2-40), (vis.width//2-20, vis.height//2-40), (vis.width//2+20, vis.height//2-40), -30, 30)

            # Mostrar frame con OpenCV (opcional)
            cv2.imshow("MediaPipe Pose", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()