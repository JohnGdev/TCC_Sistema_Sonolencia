from types import NoneType
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import math
import numpy as np
import pygame
from pathlib import Path

path = str(Path.cwd()) + "\\TCC_Sistema_Sonolencia"

def is_none(cords : list):
    try:
        for cord in cords:
            print(cord[0])
        return False
    except:
        return True

def _map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def main():
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FPS, 25.0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(height)

    closed_frames = 0
    limiar_frames = 25

    pygame.init()
    alarm_sound = pygame.mixer.Sound(path + '\\alarm.wav')

    cv2.namedWindow("MediaPipe Face Mesh", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("MediaPipe Face Mesh",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cor = (0, 0, 255)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if (results.multi_face_landmarks):
                face_landmarks = results.multi_face_landmarks[0].landmark
                cord1 = _normalized_to_pixel_coordinates(
                    face_landmarks[159].x, face_landmarks[159].y, width, height)
                cord2 = _normalized_to_pixel_coordinates(
                    face_landmarks[145].x, face_landmarks[145].y, width, height)
                cord3 = _normalized_to_pixel_coordinates(
                    face_landmarks[33].x, face_landmarks[33].y, width, height)
                cord4 = _normalized_to_pixel_coordinates(
                    face_landmarks[133].x, face_landmarks[133].y, width, height)

                cord5 = _normalized_to_pixel_coordinates(
                    face_landmarks[386].x, face_landmarks[386].y, width, height)
                cord6 = _normalized_to_pixel_coordinates(
                    face_landmarks[374].x, face_landmarks[374].y, width, height)
                cord7 = _normalized_to_pixel_coordinates(
                    face_landmarks[362].x, face_landmarks[362].y, width, height)
                cord8 = _normalized_to_pixel_coordinates(
                    face_landmarks[263].x, face_landmarks[263].y, width, height)

                # Pontos da boca para detectar bocejo
                cord9 = _normalized_to_pixel_coordinates(
                    face_landmarks[0].x, face_landmarks[0].y, width, height)
                cord10 = _normalized_to_pixel_coordinates(
                    face_landmarks[17].x, face_landmarks[17].y, width, height)
                cord11 = _normalized_to_pixel_coordinates(
                    face_landmarks[61].x, face_landmarks[61].y, width, height)
                cord12 = _normalized_to_pixel_coordinates(
                    face_landmarks[291].x, face_landmarks[291].y, width, height)

                cv2.line(image, cord1, cord2, (255, 0, 0), 4)
                #cv2.line(image, cord3, cord4, (0, 0, 255), 4)

                cv2.line(image, cord5, cord6, (255, 0, 0), 4)
                #cv2.line(image, cord7, cord8, (0, 0, 255), 4)

                cv2.line(image, cord9, cord10, (0, 255, 0), 4)
                cv2.line(image, cord11, cord12, (0, 255, 0), 4)
                
                if not is_none([cord1, cord2, cord3, cord4, cord5, cord6, cord7, cord8, cord9, cord10, cord11, cord12]):
                    
                    dist1 = math.sqrt((cord1[0] - cord2[0])
                                    ** 2 + (cord1[1] - cord2[1])**2)
                    dist2 = math.sqrt((cord4[0] - cord3[0])
                                    ** 2 + (cord4[1] - cord3[1])**2)
                    # ratio1 = dist2 / (dist1 + 0.001) #essa aqui ta invertendo

                    ratio1 = dist1 / (dist2 + 0.001)

                    dist3 = math.sqrt((cord5[0] - cord6[0])
                                    ** 2 + (cord5[1] - cord6[1])**2)
                    dist4 = math.sqrt((cord8[0] - cord7[0])
                                    ** 2 + (cord8[1] - cord7[1])**2)
                    # ratio2 = dist4 / (dist3 + 0.001) #essa aqui ta invertendo
                    ratio2 = dist3 / (dist4 + 0.001)

                    dist5 = math.sqrt((cord9[0] - cord10[0])
                                    ** 2 + (cord9[1] - cord10[1])**2)
                    dist6 = math.sqrt(
                        (cord12[0] - cord11[0])**2 + (cord12[1] - cord11[1])**2)
                    ratio3 = dist6 / (dist5 + 0.001)
                    #print("ratio1: {:.2f}".format(ratio1))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, f"FRAMES COM OS OLHOS FECHADOS: {closed_frames}", (
                        10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # Arredonda ratio1 para duas casas decimais
                    cv2.putText(
                        image, f"EAR: {ratio1:.2f}", (10, 80), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Verifica se o olho está fechado
                    if ratio1 <= 0.15 or ratio2 <= 0.15:
                        closed_frames += 1
                        if closed_frames >= limiar_frames:

                            # Preencha a imagem inteira com a cor vermelha
                            image[:] = (0, 0, 255)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(image, f" Alerta de Sono", (200, 400),
                                        font, 1, (0, 255, 255), 2, cv2.LINE_AA)
                            
                            alarm_sound.play()
                    else:
                        closed_frames = 0

                    # alerta de bocejo
                    if ratio3 <= 0.8:
                        # alarm_sound.play()
                        image[:] = (0, 255, 255)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(image, f" Alerta de Bocejo", (200, 400),
                                    font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        print("alerta de bocejo")

            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.imshow('MediaPipe Face Mesh', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()


if __name__ == '__main__':
    main()
