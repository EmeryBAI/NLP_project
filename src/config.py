from functools import lru_cache
import sys, os, yaml
@lru_cache(maxsize=1)
def load_cfg(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        cfg["PROJECT_ROOT"] = os.getcwd()
    return cfg 

global_cfg_path = os.path.join(os.getcwd(), "configs/global.yaml")
common_cfg_path = os.path.join(os.getcwd(), "configs/common.yaml")

global_cfg = load_cfg(global_cfg_path)
common_config = load_cfg(common_cfg_path)
