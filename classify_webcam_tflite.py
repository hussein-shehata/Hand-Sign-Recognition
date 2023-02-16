import sys
import os
import time

# import matplotlib
import numpy as np
# import matplotlib.pyplot as plt
import copy
import cv2
import tensorflow_hub as hub

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


interpreter = tf.lite.Interpreter(model_path='Saved Tflite Models/4-mobilenet_without_the_last9Layer_model_quant_integar_ONLY.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def preprocess_Input(X):
    X = np.array(X)
        # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        X = X / input_scale + input_zero_point

    else :
        normalised_X = X.astype(input_details["dtype"])/255.0

    normalised_X = np.expand_dims(X, axis=0).astype(input_details["dtype"])
    return normalised_X


def predict(image_data):

    interpreter.set_tensor(input_details["index"], image_data)
    interpreter.invoke()

    # Get the result 
    output_data = interpreter.get_tensor(output_details["index"])[0]   #that 0 index i dont know if i should put it or not yet
    Index = np.argmax(output_data)
    max_score = output_data.max()
    res = classes[Index]
    return res, (max_score * 100)






# Feed the image_data as input to the graph and get first prediction

c = 0

cap = cv2.VideoCapture(0)

res, score = '', 0.0
i = 0
mem = ''
consecutive = 0
sequence = ''


fps = 0
start = time.time()
frame_count = 0

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    
    if ret:
        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]
        frame = cv2.resize(img_cropped, (200, 200), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        c += 1
        # image_data = cv2.imencode('.jpg', frame)[1].tostring()
        image_data = tf.keras.utils.img_to_array(frame)
        image_data = preprocess_Input(image_data)
        
        a = cv2.waitKey(1) # waits to see if `esc` is pressed
        
        if i == 4:
            res_tmp, score = predict(image_data)
            res = res_tmp
            i = 0
            if mem == res:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive == 2 and res not in ['nothing']:
                if res == 'space':
                    sequence += ' '
                elif res == 'del':
                    sequence = sequence[:-1]
                else:
                    sequence += res
                consecutive = 0
        i += 1

        #calculate the FPS
        # Increment the frame count
        frame_count += 1
        
        # Calculate the fps every 10 frames
        if frame_count % 10 == 0:
            end = time.time()
            fps = 10 / (end - start)
            start = time.time()
            frame_count = 0
        
        # Display the fps
        print("FPS: ", fps)

        cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
        cv2.putText(img, '(score = %.5f)' % (float(score)), (0,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(img, '(FPS = %.5f)' % (float(fps)), (350,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        mem = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imshow("img", img)
        img_sequence = np.zeros((200,1200,3), np.uint8)
        cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('sequence', img_sequence)
        
        if a == 27: # when `esc` is pressed
            break

# Following line should... <-- This should work fine now
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()