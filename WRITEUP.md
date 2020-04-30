# Project Write-Up

Detect people in a designated area and determine the number of people in the frame, the average time they are in the frame, and the total count. Gain important business insight using the information generated.

Command used to run the app: 
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

## Explaining Custom Layers

The process behind converting custom layers involves...
1. Build the model 
2. Creating the Custom Layer.
3. Using Model Optimizer to Generate IR Files Containing the Custom Layer.
4. Generate the Model IR Files.
5. Inference Engine Custom Layer Implementation for the Intel CPU.

Some of the potential reasons for handling custom layers are...
- Custom layers are layers that are not included in the list of known layers. If network topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.
- This allows flexibility to design and write our own custom layer which can be converted into IR for inferencing.   

Differences:- 

Edge Computing 
- Edge Computing is regarded as ideal for operations with extreme latency concerns.
- Edge Computing is better option whe there is not enough or reliable network bandwidth to send the data to the cloud.
- When the communication network’s connection to the cloud is not robust or reliable enough to be dependable.
- Applications need rapid data sampling or must calculate results with a minimum amount of delay.
- Edge Computing requires a robust security plan including advanced authentication methods and proactively tackling attacks.

Cloud Computing
- Cloud Computing is more suitable for projects and organizations which deal with massive data storage where latency is not a major concern.
- Cloud processing power is nearly limitless. Any analysis tool can be deployed at any time.
- Results may need to be widely distributed and viewed on a variety of platforms. The cloud can be accessed from anywhere on multiple devices.
- The form factor and environmental limits of some applications could increase the cost of edge computing and make the cloud more cost-effective.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

Size of the model pre-conversion - 66.7MB
Size of the model post-conversion - 64.1MB

Inference time of model pre-conversion 
- Ran inference on Tensorflow model SSD Mobilenet V2 with same input video. On average per frame inference time was 0.13s

Inference time of model post-conversion
- Ran inference on Tensorflow model SSD Mobilenet V2 with same input video. On average per frame inference time was 0.068s

Inference time of model post-conversion was almost twice as fast when compared to inference time pre-conversion. 

Model accuracy pre and post conversion was almost same. Conversion was done with default FP32 precision hence precision of model was not reduced so accuracy of model showed no difference unless we quantized it to lower precision of FP16 or INT8 accuracy won't suffer a lot. 

## Assess Model Use Cases

Use case 
- Track activity in retail or departmental store. 
- Observe factory work spaces and building entrances for activity.
- Capture and record information on the number of people for security reasons. 

Each of these use cases would be useful because...
- To track if people gathers in some space of retail where less crowd is expected for example in billing lane or back of the store.
- For security reasons some of the sentivity areas of factory or building requires continous monitoring and survillance.
- To gather stats on people visiting some facility. This data can be further used for analysis for various purpose. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- If camera is mounted close to area being monitored where proximity is less then low accuracy model deployed on edge device may be good enough for good inference. Low capacity hardware might suffice this requirement.  
- If camera is mounted high enough at top of the building then proximity is far away and focal length should be adjusted to capture large area. For such applications high accuracy is desired to distingish people from other objects. Low capacity edge hardware with good memory may suffice such requirement as higher precision is desired. 
- If camera is mounted in market place or near traffic stop then high accuracy precision along with low confidence threshold is set to capture as many people as possible in moving frame. High capacity edge hardware device with good memory may suffice such requirement.  
- Areas with low lighting condition may require low confidence threhold with higher precision to capture people in frame.


## Model Research

Converted tensorflow SSD Mobilenet V2 model into IR representation. Steps to convert model are listed below: 

- Download the model from below given link using wget command into workspace
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

- Unzip the model directory using tar -xvf command into workspace 

- Run the following command to generate IR representation. 
python mo_tf.py --input_model /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --output_dir /home/workspace/

