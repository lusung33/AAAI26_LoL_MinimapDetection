import requests
from pathlib import Path
import yaml
import json
import os
from root import ROOT

root_path = ROOT

api_path = os.path.join(root_path, 'src', 'key', 'riot_api_key.json')
with open(api_path) as f :
    key_data = json.load(f)

api_key = key_data['api_key']

if api_key is None:
    raise ValueError("API key not found")

header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": api_key
}

def get_game_info_by_match_id(match_id) :
    match_data = []
    if match_id is not None :
        url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}"

        rq = requests.get(url, headers= header)
        if rq.status_code == 200 :
            match_data = rq.json()

    return match_data

def get_champion_name(match_data) :
    if match_data is None :
        AssertionError("No match data")
    
    players = match_data['info']['participants']
    champions = []

    for p in players :
        champions.append(p['championName'])
    
    return champions

def get_class_indices(champions, yaml_path):
    """
    target_list: List of champion names you want to convert
    yaml_path: config.yaml path (예: './config.yaml')
    return: A list of indices indicating where the champion is in the class list
    """
    # YAML 파일에서 클래스 리스트 불러오기
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    class_names = config.get('names', [])

    # 각 타겟 이름에 대해 인덱스 찾기
    indices = []
    for name in champions:
        if name in class_names:
            idx = class_names.index(name)
            indices.append(idx)
        else:
            print(f"[warning] '{name}' is not in the list of classes.")
            indices.append(-1)  # 또는 예외 처리
    return indices

if __name__ == '__main__' :
    match_id = input()
    yaml_path = (os.path.join(root_path, 'data', 'replays', 'config.yaml'))
    match_data = get_game_info_by_match_id(match_id)
    champions = get_champion_name(match_data)
    indices = get_class_indices(champions, yaml_path)
