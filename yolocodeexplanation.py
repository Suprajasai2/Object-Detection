import cv2 as cv   #Library used for image processing
import time       #Various functions related to time
Conf_threshold = 0.4  #Represents the likelihood that the detected object belongs to a particular class
NMS_threshold = 0.4   #To eliminate redundant and overlapping bounding box predictions
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]  #Represents intensities of Red,Green,Blue channels

class_name = []  #Initializes an empty list to store class names
with open('classes.txt', 'r') as f: #Opens the file named "classes.txt" in read mode using a with statement
    class_name = [cname.strip() for cname in f.readlines()] #Reads all lines from the file using readlines()


# print(class_name)
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg') #Reads the YOLOv4 Tiny model architecture
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA) #Sets the preferable backend for running the network to CUDA
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)#Sets the preferable target for computation to CUDA with FP16 precision

model = cv.dnn_DetectionModel(net) #Creates an instance which is used for performing object detection using dnn
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)# processes,scales and swaps the input images


cap = cv.VideoCapture('output.mp4') #Creates a video capture object cap using OpenCV's VideoCapture class
starting_time = time.time() # Uses the time.time() function to record the current time
frame_counter = 0 #Keep track of the number of frames processed and incremented each time a new frame is processed
while True:
    ret, frame = cap.read() # reads the next frame from the video capture
    frame_counter += 1 #Incremented to keep track of the number of frames processed
    if ret == False: #No frame was read, indicating the end of the video
        break #The loop is terminated
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)#Performs object detection on the frame
    for (classid, score, box) in zip(classes, scores, boxes):#Iterates over the detected classes,scores,bounding boxes
        color = COLORS[int(classid) % len(COLORS)]#Color for drawing the bounding box is selected based on the class ID
        label = "%s" % (class_name[classid])# Class label is retrieved
        cv.rectangle(frame, box, color, 1)#Draws a rectangle
        cv.putText(frame, label, (box[0], box[1]-10),#Puts the class label text,text position,font,size,and thickness
                   cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
    endingTime = time.time() - starting_time #Mentions the end time
    fps = frame_counter/endingTime #Mentions the frame time
    # print(fps)
    cv.putText(frame, f'FPS: {fps}', (20, 50), # FPS value is displayed on the frame
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('frame', frame) #Processed frame is displayed in a window
    key = cv.waitKey(1) #Waits for a key press for 1 millisecond
    if key == ord('q'): #The loop is terminated
        break
cap.release() # Releases the resources associated with the video capture
cv.destroyAllWindows() # Closes all open windows created by OpenCV
