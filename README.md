# FacialRecognition_Cricketers

The repository implements facial recognition on still images as well as live webcam feed. The algorithm has been trained on a dataset of 15 Indian cricketers.
The following are the uses of each file:

similarity.py
Opens a live webcam feed and prompts the name of the cricketer who has the closest resemblance to the user.
A screenshot of the output is pasted below:
![Screenshot from 2021-05-20 00-07-34](https://user-images.githubusercontent.com/43683201/118866339-83108000-b8ff-11eb-934a-753b1866bfab.png)

Full demo video at: https://youtu.be/JZXiVjyOjG8

still_images.py
Applies facial recogniton on a given still image

![test_img](https://user-images.githubusercontent.com/43683201/118865907-16958100-b8ff-11eb-89c4-5e87c03403e2.jpeg)

turns to

![Video_screenshot_19 05 2021](https://user-images.githubusercontent.com/43683201/118865943-1e552580-b8ff-11eb-8fa7-920be74e1dc4.png)


enc_res.pickle
Pickle file containing the facial encodings of all the 15 players which are used for recognition

players.csv
Contains the list of players in the dataset

encode_faces.ipynb
generates unique encodings for each face

merge.ipynb
Merges and aggregates encodings from multiple images of the same person

