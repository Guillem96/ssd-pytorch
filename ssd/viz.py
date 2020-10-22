import cv2


def draw_boxes(im, boxes, class_names, scores=None, colors=None):
    scores = [None] * len(boxes) if scores is None else scores
    colors = [None] * len(boxes) if colors is None else colors

    for params in zip(boxes, class_names, scores, colors):
        _draw_box(im, *params)

    return im


def _draw_box(im, box, class_name, score=None, color=None):
    x1, y1, x2, y2 = box
    color = color if color is not None else (0, 255, 0)
    msg = class_name.capitalize()
    if score is not None:
        msg += f' [{int(score * 100)}]'

    cv2.rectangle(im, (x1, y1), (x2, y2), color=color, thickness=2)
    cv2.rectangle(im, (x1, y1 - 20), (x2, y1), color, -1)
    cv2.putText(im, msg, (x1 + 10, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX , 
                .5, (0, 0, 0), 2, cv2.LINE_AA)

    return im
