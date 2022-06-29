""" Mavi rengi  Yada kırmızı tanıdın diyelim
Bide geometrik olarak daire şeklinde olan nesnelerin tanımasını yapacaksın
Kırmızı nokta ile tanıdığın yuvarlar aynı şeyse,O noktadan su alınacak yada bırakılacak
Yani anlatmak istediğin hem havuzun geometrisinden havuzu tanıyorsun hemde havuzun renginden havuzu tanıyorsun
Bu ikisinin de olduğu cisim havuzdur mantıken O havuza su bırakıcaz

 kırmızı yuvarlakla Mavi kare üs üste geldiği zaman inilecek diyodun değil mi
 Bide o noktanın ekranın orta noktasından ne kadar uzakta olduğunu bulmamız lazım
"""
import cv2
import numpy as np

# change it with your absolute path for the image
#Threshold of the image:Kontur algılamaya geçmeden önce, aşağıdaki snippet'i kullanarak yapabileceğimiz yukarıdaki görüntüyü eşik haline getirme
image = cv2.imread("shape.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5),cv2.BORDER_DEFAULT)
ret, thresh = cv2.threshold(blur, 200, 255,cv2.THRESH_BINARY_INV)

cv2.imwrite("thresh.png",thresh)

#İki değeri, tüm konturların bir listesini ve bunların hiyerarşilerini döndüren findContours()
#yöntemini kullanarak eşiklenmiş görüntüden konturları bulacağız
contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

blank = np.zeros(thresh.shape[:2],
                 dtype='uint8')

cv2.drawContours(blank, contours, -1,(255, 0, 0), 1)  # boş bir görüntü matrisi üzerine çize (sekillerin çerveli versiyonu)

cv2.imwrite("Contours.png", blank)

"""Görüntünün ağırlık merkezini bulmak için özel formülü kullanırız:

cx = (M10 / M00 )

cy = ( M01 / M00 )

burada cx ve cy, merkez noktanın x ve y koordinatlarıdır ve M anıdır"""
for i in contours:
    approx = cv2.approxPolyDP(i, 0.01 * cv2.arcLength(i, True), True)
    # approxPolyDP()Bir görüntünün köşe sayısını elde etmek için işlevin çıktısının uzunluğunu bulmalı
    print(len(approx))

    if len(approx) > 12:
        print("Blue = circle")
        cv2.drawContours(image, [i], 0, 255, -1)  # 0 dan sonraki renk sayıları
    elif len(approx) == 4:
        print("Red = square")
        cv2.drawContours(image, [i], 0, (
        0, 0, 255))  # drawContours() Fonksiyonu kullanarak her şekle farklı bir renk de ekleyebiliriz .



    M = cv2.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 7, (0, 0, 255), -1)
        cv2.putText(image, "center", (cx - 20, cy - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    print(f"x: {cx} y: {cy}")


cv2.imwrite("image.jpg", image)
cv2.imshow("Contours.png", blank)
cv2.imshow("image.png", image)  #bir görüntüyü bir pencerede görüntülemek için kullanılır
cv2.imshow('Binary',thresh)
cv2.waitKey()