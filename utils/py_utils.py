import cv2


def rescale(image, fx, fy=None, interpolation=cv2.INTER_LINEAR):
    if fy is None:
        fy = fx
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)


def resize(image, height, width, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(image, dsize=(width, height), interpolation=interpolation)
