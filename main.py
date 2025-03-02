import cv2
import numpy as np

def detect_lines_and_rectangles(frame):
    # Convertim la grayscale pentru procesare mai rapidă
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicăm un filtru Gaussian MAI PUTERNIC pentru reducerea zgomotului
    blur = cv2.GaussianBlur(gray, (7, 7), 0)  # De la (5,5) la (7,7)

    # Aplicăm un filtru pentru a detecta doar liniile albe
    white_mask = cv2.inRange(blur, 220, 255)  # Prag mai strict, de la 200 la 220

    # Detectăm marginile cu Canny (ajustăm pragurile)
    edges = cv2.Canny(white_mask, 100, 200)  # Creștem pragul inferior la 100 (de la 50)

    # Aplicăm transformata Hough pentru a detecta liniile drepte (eliminăm zgomotul)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=80, maxLineGap=30)

    # Copie a imaginii pentru desen
    result = frame.copy()

    # Desenăm liniile detectate pe imaginea originală (verde)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Verde (BGR)

    # 🔹 **Detectarea dreptunghiurilor (fără zgomot)**
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Eliminăm obiectele foarte mici (zgomot)
        if cv2.contourArea(contour) < 500:  # Obiecte sub 500 px² sunt ignorate
            continue

        # Aproximăm conturul pentru a verifica dacă are 4 colțuri (dreptunghi)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Verificăm dacă are exact 4 colțuri (este un dreptunghi)
            cv2.drawContours(result, [approx], 0, (0, 0, 255), 3)  # Roșu (BGR)

    return result

# 📌 Inițializează camera laptopului
cap = cv2.VideoCapture(0)  # 0 = camera principală; folosește 1 sau 2 dacă ai mai multe camere

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = detect_lines_and_rectangles(frame)

    cv2.imshow("Optimized Line & Rectangle Detection", processed_frame)

    # Apasă "Q" pentru a ieși
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()