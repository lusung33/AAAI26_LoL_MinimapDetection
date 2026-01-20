import argparse
import os
import cv2
import numpy as np
from ultralytics import RTDETR
from pathlib import Path
from src.visualize_utils import get_game_info_by_match_id, get_champion_name, get_class_indices
from root import ROOT

# path
root_path = ROOT
result_dir = root_path / 'results' / 'video'
yaml_path = (os.path.join(root_path, 'data', 'synthetics', 'lol_minimap_256_sample' , 'config.yaml'))

# identify weight name
weight_name = 'aaai26_minimap_v1'

def main(video_name, game_id, save_output=True, nfps=15):

    model_path = root_path / 'results' / 'model' / weight_name / 'weights' / 'best.pt'
    model = RTDETR(model_path)
    
    # match_id = video_name[9:-3]
    match_id = game_id
    print(f'match_id : {match_id}')
    match_data = get_game_info_by_match_id(match_id)
    champions = get_champion_name(match_data)
    indices = get_class_indices(champions, yaml_path)
    
    # Define save & video paths
    output_dir = result_dir / f'{video_name}'
    video_path = root_path / 'data' / 'videos' / video_name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    # npy output folder
    os.makedirs(output_dir, exist_ok=True)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path = None
    
    if save_output:
        output_video_path = os.path.join(output_dir, f"{weight_name}_output_video.mp4")  # Change the output video name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    all_frame_champions = []
    frame_count = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % nfps == 0:
            results = model.track(frame, persist=True)
            
            result = results[0]
            boxes = result.boxes
            conf = boxes.conf
            cls_ids = boxes.cls
            mask = (conf.cpu() >= 0.5) &  (cls_ids.int().cpu().numpy()[:, None] == indices).any(axis=1)
            boxes.data = boxes.data[mask]
            filtered_boxes_xywh = boxes.xywh.cpu().numpy()
            filtered_boxes_xyxy = boxes.xyxy.cpu().numpy()
            filtered_cls = boxes.cls.cpu().numpy()
            filtered_conf = conf.cpu().numpy()
            
            frame_data = []
            for box, cls_id, conf_score in zip(filtered_boxes_xywh, filtered_cls, filtered_conf):
                x, y, _, _ = box
                frame_data.append([frame_count, int(cls_id), float(conf_score), x, y])

            all_frame_champions.extend(frame_data)
            # frame = result.plot()
            for box in filtered_boxes_xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imshow('Filtered Detection', frame)
            cv2.waitKey(nfps)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

        # if result is not None: 
            # frame = result.plot()

        if save_output:
            out.write(frame)

    # Save model output
    # all_frame_array = np.array(all_frame_champions)
    # np.save(f'{output_dir}/{weight_name}_tracking_results.npy', all_frame_array)
    
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

    if save_output:
        print(f"Output video saved at {output_video_path}")
    print(f"Output saved at {output_dir}")
    print('total frame : ', frame_count)

if __name__ == "__main__":
    '''
    python -m scripts.vis.visualizer --video_name vis_sample_video.mp4 --game_id KR_7691650081 --save --nfps 1
    '''
    parser = argparse.ArgumentParser(description="RT-DETR Video Tracking")
    parser.add_argument("--video_name", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--game_id", type= str, required= True, help="game id")
    parser.add_argument("--save", action="store_true", help="Flag to save the output video")
    parser.add_argument('--nfps', type=int, default=1, help='fps inference (default: 1)')

    args = parser.parse_args()

    main(args.video_name, args.game_id, args.save, nfps=args.nfps)