import os
import mediapipe as mp
from utils import process_img, process_video, process_webcam
import argparse
from enum import Enum, auto


class MediaType(Enum):
    IMAGE = auto(),
    VIDEO = auto(),
    WEBCAM = auto()


def blur_faces(media_type: MediaType, file_path: str, output_dir: str, camera_id: int = 0):

    # get file name
    file_name = os.path.basename(file_path)

    # detect faces object
    mp_face_detection = mp.solutions.face_detection

    # model_selection: 0 for faces close to the camera, 2 meters
    # model_selection: 1 for faces far from the camera, > 2 meters < 5 meters
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        match media_type:
            case MediaType.IMAGE:
                process_img(file_path, file_name, output_dir, face_detection)
            case MediaType.VIDEO:
                process_video(file_path, file_name, output_dir, face_detection)
            case MediaType.WEBCAM:
                process_webcam(face_detection, cam_id=camera_id)
            case _:
                print('Unknown media type.')


def validate_media_type(media_type_str):
    try:
        media_type_enum = MediaType[media_type_str.upper()]
        return media_type_enum
    except KeyError:
        return None


if __name__ == '__main__':
    OUTPUT_DIR = './output'
    CAMERA_ID = 0

    try:
        # initialize args
        parser = argparse.ArgumentParser(prog='FaceBlocker', description="Hides the mug.")
        parser.add_argument("--mode", default='video')
        parser.add_argument("--filePath", default='./data/testVideo.mp4')

        # get args
        args = parser.parse_args()

        validated_media_type = validate_media_type(args.mode)

        if validated_media_type:
            # create output dir
            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)

            # blur the faces
            blur_faces(media_type=validated_media_type,
                       file_path=args.filePath,
                       output_dir=OUTPUT_DIR,
                       camera_id=CAMERA_ID)
        else:
            print(f'Input is not valid:\nMode:{args.mode}\nPath:{args.filePath}')
    except Exception as e:
        print(f'Welp, something is wrong: {e}')

