## Directory Structure
```bash
MinimapDetection
├── assets/
├── data/
│   ├── replays. # replay dataset
│   ├── synthetics. # synthetic dataset
│   └── videos. # sample video for visualize (not uploaded)
│
├── results/ 
│   ├── eval/ # model eval results
│   ├── model/ # save path for model training (pretrain, finetune)
│   └── video/ # visualizing results
│
├── scripts/
│   ├── data/
│   ├── eval/
│   ├── train/
│   └── vis/
│
├── src/
│   └── key/ # riot_api_key.json needed
│
├── root
├── .gitattributes
├── .gitignore
├── Dockerfile
├── environment.yml
└── README.md
```

## Setting
```bash
# git clone
git clone git@github.com:lusung33/MinimapDetection.git # need sshkey gen

# move to folder
cd MinimapDetection

# pull or build
docker build -t <yourID>/<yourTAG>:v1 . # directly build

# docker container execute
docker run -it --gpus all --shm-size=8g --name esports -v /mnt/nas/esports/minimap_detection/synthetic_images:/home/<yourID>/MinimapDetection/data/synthetics <yourID>/<yourTAG>:v1
```

If you train a model, setup the datapath in `setup_ultralytics.py`and run. (Only if ultralytics dataset_dir error occured.)

```bash
# setting
python setup_ultralytics.py
```

## Dataset
The dataset is available at the following NAS path.
| 데이터셋 이름 | NAS 경로          | 옮길 경로       |
|---------------|------------------|------------|
| Synthetic Images | /esports/minimap_detection/synthetic_images/lol_minimap_256_n00k | data/synthetics/lol_minimap_256_n00k |
| Replay Minimap Videos | /esports/minimap_detection/replay_dataset_170 | /data/replays/replay_dataset_170 |

## Train Model
```bash
# pretrain RT-DETR model
python scripts.train.model_train --file_name lol_minimap_256_n00k --epoch 100 --batches 32 --imgsz 256 # set params

# finetuning RT-DETR model
python -m scripts.train.model_finetune --model <pretrain output model name> # set params
```

## Visualization
```bash
# set weight path & video path in the code directly before run
python -m scripts.vis.visualization --video_name 256_KR_7586158292.mp4 --game_id KR_7586158292 --model_name rtdetr --save --nfps 1
```
## Evaluation
```bash
# set weight path in the code directly before run
python -m scripts.eval.evaluation
```