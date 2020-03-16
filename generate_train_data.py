import os
import cv2
import dlib
import time
import argparse
import numpy as np
from imutils import video

DOWNSAMPLE_RATIO = 4


def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def main():
    os.makedirs('original', exist_ok=True)
    os.makedirs('landmarks', exist_ok=True)

    cap = cv2.VideoCapture(args.filename)
    fps = video.FPS().start()

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        t = time.time()

        # Perform if there is a face detected
        if len(faces) >= 0:


            # Display the resulting frame
            count += 1
            print(count)
            #cv2.imwrite("original/{}.png".format(count), frame)
            #cv2.imwrite("landmarks/{}.png".format(count), black_image)
            cv2.imshow('a',frame_resize)
            fps.update()

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

            if count == args.number:  # only take 400 photos
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No face detected")

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
    parser.add_argument('--num', dest='number', type=int, help='Number of train data to be created.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
