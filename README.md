# anime-face-detector

「Anime-face-detector」は機械学習を用いてアニメの顔検出を可能にします。これはRetina faceのTensorflow実装（[retinaface-tf2](https://github.com/peteryuX/retinaface-tf2 "retinaface-tf2")）を用いており、トレーニングは「[Tagged Anime Illustrations](https://www.kaggle.com/mylesoneill/tagged-anime-illustrations "Kaggle Tagged Anime Illustrations")」のデータセットを用いました。  
Anime-face-detector is a machine-learning based face detect engine in Python, using retinaface-tf2, Kaggle Tagged Anime Illustrations.

## Basic Usage
> python detect_anime_face.py --cfg_path="./configs/retinaface_mbv2.yaml" --img_path="./test.jpg" --down_scale_factor=1.0

## For more details
- [About precision](https://hanadamaya.net/face_detection_anime2/)
- [About Maya Hanada](https://twitter.com/Hanada_Maya)