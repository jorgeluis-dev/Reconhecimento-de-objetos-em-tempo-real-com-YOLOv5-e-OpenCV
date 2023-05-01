import cv2
import torch

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Definir as classes de objetos que queremos detectar
classes = ['person', 'car', 'truck']

# Definir as cores para cada classe
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Ler o próximo quadro do vídeo
    ret, frame = cap.read()

    # Detectar objetos no quadro
    results = model(frame, size=640)

    # Extrair as informações dos objetos detectados
    objects = results.pandas().xyxy[0]

    # Desenhar caixas delimitadoras e rótulos para cada objeto detectado
    for index, obj in objects.iterrows():
        x1, y1, x2, y2 = obj[['xmin', 'ymin', 'xmax', 'ymax']].values
        cls = int(obj['class'])
        if cls < len(classes):
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls], 2)
            cv2.putText(frame, classes[cls], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cls], 2)

    # Exibir o quadro com os objetos detectados
    cv2.imshow('Object Detection', frame)

    # Verificar se o usuário pressionou a tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar as janelas
cap.release()
cv2.destroyAllWindows()
