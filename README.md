# EfficientAD implementation
inspired by:
https://github.com/nelson1425/EfficientAD and https://github.com/rximg/EfficientAD/tree/main

original paper: https://arxiv.org/pdf/2303.14535v2.pdf

## setup

### dataset
download mvtec ad:
```
cd datasets
mkdir mvtec_ad
cd mvtec_ad
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz
rm mvtec_anomaly_detection.tar.xz
cd ../..
```

download mvtec loco:
```
cd datasets
mkdir mvtec_loco
wget https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz
tar -xvf mvtec_loco_anomaly_detection.tar.xz
rm mvtec_loco_anomaly_detection.tar.xz
cd ../..
```