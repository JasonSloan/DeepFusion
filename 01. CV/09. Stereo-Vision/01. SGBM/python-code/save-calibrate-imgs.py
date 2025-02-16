import cv2
import collections
from pathlib import Path

Camera = collections.namedtuple("Camera", ["index", "name"])


def show_camera_imgs(camera1, camera2):
    cap1 = cv2.VideoCapture(camera1.index)
    cap2 = cv2.VideoCapture(camera2.index)
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        cv2.imshow(camera1.name, frame1)
        cv2.imshow(camera2.name, frame2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


def save_calibrate_imgs(camera1, camera2, save_dir):
    save_dir = Path(save_dir)
    save_dir1 = save_dir / camera1.name
    save_dir2 = save_dir / camera2.name
    save_dir1.mkdir(parents=True, exist_ok=True)
    save_dir2.mkdir(parents=True, exist_ok=True)

    count = 0
    cap1 = cv2.VideoCapture(camera1.index)
    cap2 = cv2.VideoCapture(camera2.index)
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        cv2.imshow(camera1.name, frame1)
        cv2.imshow(camera2.name, frame2)

        key = cv2.waitKey(1)
        if key == ord(" "):
            cv2.imwrite(save_dir1 / f"{camera1.name}_{count}.jpg", frame1)
            cv2.imwrite(save_dir2 / f"{camera2.name}_{count}.jpg", frame2)
            count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera1 = Camera(index=0, name="left")
    camera2 = Camera(index=2, name="right")
    # show_camera_imgs(camera1, camera2)
    save_calibrate_imgs(camera1, camera2, save_dir="calibrate/imgs")
