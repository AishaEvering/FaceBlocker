import cv2
import os


def process_img(path: str, name: str, output_dir: str, face_detection) -> None:
    """
    Processing face detection data, adding blur, and saving the output.
    :param path: Path to image
    :param name: Name of image
    :param output_dir: Path where the processed image will be saved
    :param face_detection: face detection obj
    :return: Boolean that indicates success/failure
    """
    try:
        # read image
        img = cv2.imread(path)
        # process image
        img = __process_img(img, face_detection)
        # save image
        cv2.imwrite(os.path.join(output_dir, f'blurred_{name}'), img)

    except Exception as e:
        print(f'Unable to process image: {e}')


def process_video(path: str, name: str, output_dir: str, face_detection) -> None:
    """
    Processing face detection data, adding blur, and saving the output.
    :param path: Path to video
    :param name: Name of video
    :param output_dir: Path where the processed video will be saved
    :param face_detection: face detection obj
    :return: Boolean that indicates success/failure
    """
    try:
        # read video
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()

        (height, width) = frame.shape[:2]  # Extract height and width

        # init video writer
        output_video = cv2.VideoWriter(os.path.join(output_dir, f'blurred_{name}'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       # 4-character code of codec used to compress the frames
                                       cap.get(cv2.CAP_PROP_FPS),  # get frames per second
                                       (width, height))  # Frame size (width, height)
        while ret:
            # process video
            frame = __process_img(frame, face_detection)
            # save video frame by frame
            output_video.write(frame)
            # read next frame
            ret, frame = cap.read()

        cap.release()
        output_video.release()

    except Exception as e:
        print(f'Unable to process video: {e}')


def process_webcam(face_detection, cam_id: int = 0) -> None:
    """
    Processing face detection data, adding blur, and saving the output.
    :param face_detection: face detection obj
    :param cam_id: id of web camera
    :return: Boolean that indicates success/failure
    """
    try:
        # read video
        cap = cv2.VideoCapture(cam_id)
        ret, frame = cap.read()

        while ret:
            # process video
            frame = __process_img(frame, face_detection)
            # show updated feed
            cv2.imshow('frame', frame)
            cv2.waitKey(25)  # pause
            # read next frame
            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    except Exception as e:
        print(f'Unable to process webcam: {e}')


def __process_img(img, face_detection):
    """
    Finds the faces and blurs the faces
    :param img: image that will be processed
    :param face_detection: face detection obj
    :return: processed image, image with the faces blurred
    """
    height, width, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting from BGR to RGB
    out = face_detection.process(img_rgb)

    if out.detections:  # if face found
        for detection in out.detections:  # get all the bounding boxes
            bbox = detection.location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * width)
            y1 = int(y1 * height)
            w = int(w * width)
            h = int(h * height)

            # blur faces
            # thinking of the image shape we are blurring form the point
            # of y1 to the y1 + height (value from face detection) and the
            # same with x
            # the channels remain the same
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (40, 40))

    return img
