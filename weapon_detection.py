import cv2
import numpy as np

# Let's load the YOLO model - this is the brain of our weapon detection system!
# You'll need to grab the weight file (yolov3_training_2000.weights) from this Google Drive link: 
# https://drive.google.com/file/d/10uJEsUpQI3EmD98iwrwzbD4e19Ps-LHZ/view?usp=sharing
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")

# We're only looking for weapons, so let's keep it simple with just one class
classes = ["Weapon"]

# This part is commented out, but you could use it to load more classes from a file if needed
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.read()]

# Getting the output layers where YOLO will give us the detection results
output_layer_names = net.getUnconnectedOutLayersNames()

# Generating some random colors for the bounding boxes - makes it look nice and colorful!
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Let's create a function to let the user choose a video file or use the webcam
def value():
    val = input("Enter a video file name (like 'ak47.mp4') or just hit Enter to use your webcam: \n")
    if val == "":
        val = 0  # 0 means webcam
    return val

# Start capturing video from the chosen source (file or webcam)
cap = cv2.VideoCapture(value())

# Here comes the main loop where the magic happens!
while True:
    # Grab a frame from the video
    success, img = cap.read()
    if not success:
        print("Oops! Couldn't read a frame from the video. Something went wrong.")
        break

    # Get the dimensions of the frame
    height, width, channels = img.shape

    # Prepare the image for YOLO: resize it to 416x416 and convert to a blob
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Feed the blob into the network and get the detection results
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Lists to store detected objects' details
    class_ids = []
    confidences = []
    boxes = []

    # Process each detection from the network output
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Scores for each class
            class_id = np.argmax(scores)  # Pick the class with the highest score
            confidence = scores[class_id]  # How confident is YOLO about this detection?
            
            # If we're confident enough (above 50%), let's mark it as a detection
            if confidence > 0.5:
                # Calculate the bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Save the box, confidence, and class ID
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Use Non-Max Suppression to filter out overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # If we detected something, let the user know!
    if len(indexes) > 0:
        print("Weapon detected in frame!")

    # Set up the font for labeling the detected objects
    font = cv2.FONT_HERSHEY_PLAIN

    # Draw boxes and labels on the image for each detection
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Get the class name (Weapon)
            color = colors[class_ids[i]]  # Pick a color for this class
            # Draw a rectangle around the detected object
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Add the label above the box
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # Show the processed frame with detections
    cv2.imshow("Image", img)

    # Press 'Esc' (key code 27) to exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Clean up: release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
