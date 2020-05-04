"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

#INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

    

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
        
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    if args.cpu_extension:
        infer_network.load_model(args.model, args.device, args.cpu_extension)
    else:
        infer_network.load_model(args.model, args.device, CPU_EXTENSION)
    
    ### TODO: Handle the input stream ###
    net_input_shape = infer_network.get_input_shape()
    
    ## Get and open video capture
    image_flag = False
    
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('jpg') or args.input.endswith('bmp'):
        image_flag = True
    elif args.input.endswith('mp4'):
        image_flag = False
    else:
        print("Unsupported input, valid inputs are image(jpg and bmp), video file(mp4) or webcam/video stream.")    
        
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Create a video writer for the output video
    if image_flag:
        out = None  
    else:
        out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    total_count = 0
    current_count = 0 
    last_count = 0
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
            
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        
        #inf_start = time.time()
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.async_inference(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            #det_time = time.time() - inf_start
            #print(det_time)
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.extract_output()
            
            ### TODO: Extract any desired stats from the results ###
            out_frame, current_count = processed_and_create_output(result, frame, prob_threshold)
            
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                
                 # Publish "person/total_count" messages to the MQTT server
                client.publish("person", json.dumps({"total": total_count}))
            
            if current_count < last_count and int(time.time() - start_time) >=1: # added lag of 1sec to avoid flickering and unstability
                duration = int(time.time() - start_time)
                
                # Publish "person/duration" messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            # Publish "person/current_count" messages to the MQTT server    
            client.publish("person", json.dumps({"count": current_count}))
            
            last_count = current_count
            
                  
            

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_img.jpg', out_frame)
        
        # Break if escape key pressed
        if key_pressed == 27:
            break
        
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Disconnect from MQTT
    client.disconnect()
    


def processed_and_create_output(output, input_img, threshold):

    height = input_img.shape[0]
    width = input_img.shape[1]
    
    current_count = 0
    
    for box in output[0][0]:
        
        if box[2] > threshold:
            #box_lst.append(box)
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width) 
            ymax = int(box[6] * height)

            cv2.rectangle(input_img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            
            current_count = current_count + 1
            #current_count = 1
    
    
    return input_img, current_count
    
        
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    
    
if __name__ == '__main__':
    main()
