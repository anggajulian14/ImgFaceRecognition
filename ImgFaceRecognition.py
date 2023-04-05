import cv2
import face_recognition

img = cv2.imread("images/Indonesia 7th President.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("images/Indonesia Speaker of Parliament.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

width, height = 500,500
resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
resized_img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)

result = face_recognition.compare_faces([img_encoding], img_encoding2)

num_matches = sum(result)
total_comparation = len(result)
percentage = (num_matches / total_comparation ) * 100
print("Percentage similiartity : {:.2f}%".format(percentage))

cv2.imshow("Img", resized_img)
cv2.imshow("Img 2", resized_img2)
cv2.waitKey(0)