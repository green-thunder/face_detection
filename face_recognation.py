import face_recognition as fr
import cv2
import numpy as np
import os
path = "images/"

known_names = []
known_name_encodings = []

images = os.listdir(path)

for img in images:
    image = fr.load_image_file(path + img)
    image_path = path + img
    encoding = fr.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())
        
print(known_names)

vid = cv2.VideoCapture(0) 
name = ""

while True:
    ret, frame = vid.read()

    if not ret:
        print("Error reading frame")
        break

    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)
      
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        
        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
        else:
            name = "Unknown"  

        # cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        print(name)
    cv2.imshow('Facial Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
cv2.destroyAllWindows() 
