# Code and Dataset for Training Diagram Parser in AutoGPS

This repository provide our training code for the diagram parser used in [AutoGPS demo](https://github.com/Jayce-Ping/AutoGPS/).

## Dependencies

- Run the following command to setup the dependencies (create a virtual enviroment in advance if needed):

```bash
bash setup.sh
```

## Train

- Run the following command to train a U-Net model:
```bash
python train-unet.py --dataset_dir datasets/PGDP5K_yolo_seg\
    --batch_size 32
```


- Runt he following command to train a YOLO model:
```bash
python train-yolo.py --model_name yolo11n-seg
```

The best model `best-unet.pth` and `best-yolo11n-seg.pt` will be saved in `models` folder. These models are used in [AutoGPS-Demo](https://github.com/Jayce-Ping/AutoGPS/tree/main/demo).