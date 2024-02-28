import cv2
import mediapipe as mp

# Inicialize o objeto FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Inicialize a captura de vídeo
cap = cv2.VideoCapture(1)  # 0 representa a câmera padrão, pode ser ajustado se você tiver várias câmeras

while cap.isOpened():
    # Leia um quadro do vídeo
    ret, frame = cap.read()
    if not ret:
        break

    # Converta o quadro para o formato RGB (MediaPipe usa RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Execute a inferência
    resultado = face_mesh.process(frame_rgb)

    # Verifique se há pontos no rosto e extraia se existirem
    if resultado.multi_face_landmarks:
        for landmarks in resultado.multi_face_landmarks:
            for ponto_id, ponto in enumerate(landmarks.landmark):
                altura, largura, _ = frame.shape
                x, y = int(ponto.x * largura), int(ponto.y * altura)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Reduza o valor do raio (de 5 para 2)

    # Exiba o frame com os pontos do rosto
    cv2.imshow("Pontos do Rosto", frame)

    # Saia do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos quando terminar
cap.release()
cv2.destroyAllWindows()
