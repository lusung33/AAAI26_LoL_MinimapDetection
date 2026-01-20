from ultralytics import RTDETR
from pathlib import Path
import argparse
from root import ROOT

root_path = ROOT
data_path = root_path / 'data' / 'replays' / 'config.yaml'
result_dir = root_path / 'results' / 'model' / 'finetune'

def main(model_name) :
    weights_path = root_path / 'results' / 'model' / 'pretrain' / f'{model_name}' / 'weights' / 'best.pt'
    # 이전 학습된 모델 불러오기
    model = RTDETR(weights_path)

    # 이어서 파인튜닝
    model.train(
        data= data_path,
        imgsz= 256,
        epochs= 100,
        patience= 20,

        batch= 8,
        device= 0,
        workers= 4,
        
        lr0= 0.00005,
        lrf= 0.05,
        weight_decay= 0.00005,
        optimizer= 'AdamW',
        cos_lr= True,
        warmup_bias_lr= 0.0005,
        
        hsv_h= 0.01,
        hsv_s= 0.2,
        hsv_v= 0.1,
        degrees= 0,
        translate= 0,
        scale= 0,
        shear= 0,
        perspective= 0,
        flipud= 0,
        fliplr= 0,
        mosaic= 0,
        erasing= 0,
        auto_augment= None,

        name= f'{model_name}_finetuned',
        project= result_dir,
    )

# 실행 예시
# python -m scripts.train.model_finetune
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="Train the model with the given parameters.")
    parser.add_argument('--model', type= str, required= True, help= "Name of the model")
    args = parser.parse_args()

    main(args.model)
