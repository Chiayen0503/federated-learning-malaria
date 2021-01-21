# federated-learning-malaria
Data preparation and weight averaging example codes

I trained a yolov5 model to identify malaria. However, to apply Federated learning on yolov5, I wrote a python file to do weight averaging on client models. 

In addition, due to raw dataset (malaria) only provided .json files and had some other irrelevant classes, I did some preprocessing to generate labels that fit yolov5 format. 

There are few variables you can modify in weight_averaging.py

    (1) weight_1_path & weight_2_path: client weight paths

    (2) config: derives from the yolov5 model you choose (yolov5l.yaml, yolov5m.yaml, yolov5s.yaml, yolov5x.yaml)

    (3) channel: 3 (RGB), 2(grayscale) 

    (4) beta: (client datasize/total) for each client 

Some problems have not been fixed:

(1) unbalanced data
(2) testing on the averaging weight ("avg.pt") got a poor result. It might be because client models didn't converage to same local minimum. 
(3) we have not retrained avg.pt with server data


Data source: https://bbbc.broadinstitute.org/BBBC041

YoloV5: https://github.com/ultralytics/yolov5
