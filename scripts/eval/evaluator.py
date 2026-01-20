import os
from ultralytics import RTDETR, settings
from pathlib import Path
from root import ROOT

def main():
    root_path = ROOT

    weight_dir = root_path / 'results' / 'model'
    dataset_dir = root_path / 'data' / 'synthetics' / 'lol_minimap_256_sample'
    output_dir = root_path / 'results' / 'eval'

    settings.update({"datasets_dir": str(root_path)})
    dataset_config = os.path.join(dataset_dir, 'config.yaml')

    # Evaluate the models
    results_list = []

    # Weight pathes
    weight_path = weight_dir / 'aaai26_minimap_v1' / 'weights' / 'best.pt'
    print(weight_path)

    if os.path.exists(weight_path):
        model = RTDETR(weight_path)
        metrics = model.val(data= dataset_config, project= output_dir, name= f"aaai26_minimap_v1", split= "test")
        map_5095 = metrics.box.map
        print((i, map_5095, weight_path))
        results_list.append((i, map_5095, weight_path))

    top_sort = sorted(results_list, key=lambda x: x[1], reverse=True)[:5]

    for rank, (i, map_5095, path) in enumerate(top_sort, 1):
        print(f"Top_sort{rank}: train{i} | mAP@0.5:0.95 = {map_5095:.4f} | {path}")

# 실행 예시
# python -m scripts.eval.evaluator
if __name__ == '__main__':

    main()
