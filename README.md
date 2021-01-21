# federated-learning-malaria
data preparation and weight averaging example codes

I trained a yolov5 model to identify malaria classes. However, to apply Federated learning on yolov5, I wrote a python file to do weight averaging on client models. 

In addition, due to raw dataset (malaria) only provided .json files and had some other irrelevant classes, I did some preprocessing to generate labels that fit yolov5 format. 

There are few variables you can modify in weight_averaging.py

(1) weight_1_path & weight_2_path: client weight paths

(2) config: derives from the yolov5 model you choose (yolov5l.yaml, yolov5m.yaml, yolov5s.yaml, yolov5x.yaml)

(3) channel: 3 (RGB), 2(grayscale) 

(4) beta: (client datasize/total) for each client 
