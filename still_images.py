import cv2
import numpy as np
import face_recognition as fr
import matplotlib.pyplot as plt
import pickle

IMG_PATH = 'test_img.jpeg'

#Loading encodings
file = open('enc_res.pickle', 'rb')
known_faces = pickle.load(file)
file.close()
known_face_names = list(known_faces.keys())
known_face_enc = list(known_faces.values())

face_locations = []
face_encodings = []
face_names = []

final_frame = cv2.imread(IMG_PATH)
# resize_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
# final_frame = resize_frame[:, :, ::-1]
face_locations = fr.face_locations(final_frame)
face_encodings = fr.face_encodings(final_frame, face_locations)
face_names = []
for face_encoding in face_encodings:
    matches = fr.compare_faces(known_face_enc, face_encoding, tolerance=0.8)
    name = "Unknown"
    face_distances = fr.face_distance(known_face_enc, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    print(name)
    face_names.append(name)
for (top, right, bottom, left), name in zip(face_locations, face_names):
    
    cv2.rectangle(final_frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(final_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(final_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
cv2.imshow('Video', final_frame)
cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()

