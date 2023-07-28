import cv2
import os


# TODO faces from image decoder

# def detect_and_crop_faces(image_path):
#     image = cv2.imread(image_path)
#
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for i, (x, y, w, h) in enumerate(faces):
#         cropped_face = image[y:y + h, x:x + w]
#
#         cv2.imshow("Cropped Face", cropped_face)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         output_file = f"temp/cropped_face_{i + 1}.jpg"
#         cv2.imwrite(output_file, cropped_face)


# TODO faces from video decoder


def detect_and_crop_faces_from_video(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    video_capture = cv2.VideoCapture(video_path)

    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    os.makedirs('output_folder', exist_ok=True)

    frame_count = 0

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % fps == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for i, (x, y, w, h) in enumerate(faces):
                cropped_face = frame[y:y + h, x:x + w]

                output_file = os.path.join('output_folder', f"cropped_face_{frame_count}_{i + 1}.jpg")
                cv2.imwrite(output_file, cropped_face)

    video_capture.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_crop_faces_from_video('temp/rickroll.mp4')

if __name__ == "__main__":
    # TODO faces from image decoder

    # input_image_path = "temp/men.jpg"
    # detect_and_crop_faces(input_image_path)

    pass
