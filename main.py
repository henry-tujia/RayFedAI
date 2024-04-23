"""
Author: henry tanhao0606@outlook.com
Date: 2024-04-17 13:02:38
LastEditors: henry tanhao0606@outlook.com
LastEditTime: 2024-04-17 13:07:37
FilePath: /Multi-FL-Training-main/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import argparse
import pathlib
import datetime
from typing import Optional
import rootutils

from omegaconf import DictConfig
import omegaconf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True, cwd=True)

from src.train import Trainer
from src.utils import tools

omegaconf.OmegaConf.register_new_resolver("uuid", tools.generate_uuid)


import hydra
# import os

# os.environ['http_proxy'] = "http://127.0.0.1:7890"
# os.environ['https_proxy'] = "http://127.0.0.1:7890"


def main():
    dt = datetime.datetime.now()

    parser = argparse.ArgumentParser(description="path to get yaml.")
    parser.add_argument(
        "--method",
        type=str,
        default="dynafed",
        help="fedavg or others",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str("./yamls/base.yaml"),
        help="the path of yaml",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="debug",
        help="task name for recognize",
    )

    parser.add_argument("--tags", nargs="+", default=[""])

    args = parser.parse_args()

    SAVE_PATH = pathlib.Path("./save") / args.task / f"""{dt.isoformat()}"""

    pathlib.Path.mkdir(SAVE_PATH, exist_ok=True, parents=True)

    trainer = Trainer(args.method, pathlib.Path(args.config), SAVE_PATH, args.tags)
    trainer.run()


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main_hyder(cfg: DictConfig) -> Optional[float]:
    logger = tools.set_logger(cfg.paths.output_dir, "scheduler")
    tools.print_config_tree(cfg, loggger=logger, save_to_file=True, resolve=True)
    cfg.paths.output_dir = pathlib.Path(cfg.paths.output_dir)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main_hyder()
