import cv2
import face_recognition
import yt_dlp
import numpy as np
# import json
# model = YOLO("yolov8n.pt")  # load an official detection model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
link = input("Enter the link: ")
# class VideoStream:
#     url: str = None
#     resolution: str = None
#     height: int = 0
#     width: int = 0
#
#     def __init__(self, video_format):
#         self.url = video_format['url']
#         self.resolution = video_format['format_note']
#         self.height = video_format['height']
#         self.width = video_format['width']

if("youtu" in link):
    #https://www.youtube.com/shorts/ciaGj27dcIk testing
    ydl_opts = {'format': "bestvideo[height<=1080]+bestaudio/best[height<=1080]","merge_output_format": 'mp4'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        # info = ydl.download(link)
        # streams = [VideoStream(format)
        #            for format in info['formats'][::-1]
        #            if format['vcodec'] != 'none' and 'format_note' in format]
        # _, unique_indices = np.unique(np.array([stream.resolution
        #                                         for stream in streams]), return_index=True)
        # streams = [streams[index] for index in np.sort(unique_indices)]
        # resolutions = np.array([stream.resolution for stream in streams])
    input_mp4 = cv2.VideoCapture(info['formats'][-1]['url'])
else:
    try:
        input_mp4 = cv2.VideoCapture(link)
    except Exception as e:
        print(e)
        exit(1)

length = int(input_mp4.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 29.97, (info['formats'][-1]['width'],info['formats'][-1]['height']))
# cv2.namedWindow('video', cv2.WINDOW_NORMAL)
try:
    face = face_recognition.load_image_file(input("Type Face File location")) #"Screenshot 2024-06-17 143922.png"
    face_encoding = face_recognition.face_encodings(face)[0]
except Exception as e:
    print(e)
    exit(1)

known_faces = [
    face_encoding]

face_locations = []
face_encodings = []
face_names = []

frame_number = 0
num_frames = 20

while True:
    # Grab a single frame of video
    ret, frame = input_mp4.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break
    # cv2.imshow('video', frame)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_frame = frame[:, :, ::-1] this one caused error
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Prime"
            print("found")

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output.write(frame)
# All done!
input_mp4.release()
cv2.destroyAllWindows()