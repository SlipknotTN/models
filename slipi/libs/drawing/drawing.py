import numpy as np


def draw_masks_on_single_image(predictions, classes, threshold=0.7):

    detections = []

    # Filter by score
    for index in range(predictions["num_detections"]):
        score = predictions["detection_scores"][index]
        if score > threshold:
            detection_mask = predictions["detection_masks"][index]
            # In case of face prediction we need to put it at the lowest level
            # in case of multiple prediction on the same place
            class_name = classes[predictions["detection_classes"][index]]
            # Class color in RGB format
            class_color = np.array(classes[predictions["detection_classes"][index]][1])
            if class_name == "face":
                detections.insert(0, (detection_mask, class_color))
            else:
                detections.append((detection_mask, class_color))

    allinone_prediction = np.zeros((predictions["detection_masks"].shape[1],
                                    predictions["detection_masks"].shape[2], 3), dtype=np.uint8)

    # Apply background color
    allinone_prediction[::] = classes[0][1]

    for detection in detections:
        allinone_prediction[detection[0] == 1] = detection[1]

    return allinone_prediction