import cv2
import numpy as np


image = cv2.imread("shape.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5),cv2.BORDER_DEFAULT)
ret, thresh = cv2.threshold(blur, 200, 255,cv2.THRESH_BINARY_INV)
#Threshold of the image:Kontur algılamaya geçmeden önce, aşağıdaki snippet'i kullanarak yapabileceğimiz yukarıdaki görüntüyü eşik haline getirme


cv2.imwrite("../SekilRenkAlgilama/thresh.png", thresh)
#İki değeri, tüm konturların bir listesini ve bunların hiyerarşilerini döndüren findContours()
#yöntemini kullanarak eşiklenmiş görüntüden konturları bulacağız
contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

blank = np.zeros(thresh.shape[:2],dtype='uint8')

cv2.drawContours(blank, contours, -1,(255, 0, 0), 1)  # boş bir görüntü matrisi üzerine çize (sekillerin çerveli versiyonu)

cv2.imwrite("Contours.png", blank)

"""Görüntünün ağırlık merkezini bulmak için özel formülü kullanırız:
cx = (M10 / M00 )
cy = ( M01 / M00 )
burada cx ve cy, merkez noktanın x ve y koordinatlarıdır ve M anıdır"""
for i in contours:

    approx = cv2.approxPolyDP(i, 0.01 * cv2.arcLength(i, True), True)
    # approxPolyDP()cv2 (yaklaşıkPolyDP) nedir?
    # cv2.approxPolyDP ile ilgili görsel sonucu
    # Belirli bir çokgenin bir konturunun şeklini,
    # orijinal çokgenin şekline belirtilen kesinlikte yaklaştırma işlemine , kontur şeklinin yaklaşımı denir
    print(f"approx: {len(approx)}")

    M = cv2.moments(i)
    if M['m00'] != 0: #merkez bulma kısmı
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(image, "center", (x - 20, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    print(f"x: {x} y: {y}")

    # putting shape name at center of each shape(şekillere isim verme)
    if len(approx) == 3:
        cv2.putText(image, 'Triangle', (x - 40, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    elif len(approx) == 4:
        cv2.putText(image, 'Quadrilateral', (x - 40, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    elif len(approx) == 5:
        cv2.putText(image, 'Pentagon', (x - 40, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    elif len(approx) == 6:
        cv2.putText(image, 'Hexagon', (x - 40, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    else:
        cv2.putText(image, 'circle', (x - 40, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


cv2.imwrite("image.jpg", image)
cv2.imshow("Contours.png", blank)
cv2.imshow("image.png", image)  #bir görüntüyü bir pencerede görüntülemek için kullanılır
cv2.imshow('Binary',thresh)
cv2.waitKey()