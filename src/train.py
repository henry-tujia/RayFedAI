from collections import defaultdict
import time
import logging
from omegaconf import DictConfig, OmegaConf
import rich
import tqdm
import random
import ray
import pathlib
import torch
from torch.multiprocessing import Queue, set_start_method
from torch.utils.tensorboard import SummaryWriter
import wandb
# from aim import Run

from src.utils import custom_multiprocess, tools
from src.models.init_model import Init_Model

torch.multiprocessing.set_sharing_strategy("file_system")


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        DEVICE = (
            torch.device(cfg.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = DEVICE
        self.logger_method = tools.set_logger
        self.logger = self.logger_method(cfg.paths.output_dir, "train")
        self.writer = self.setup_tensorboard()

    def setup_tensorboard(self):
        save_path = self.cfg.paths.output_dir / "tensorboard"
        pathlib.Path.mkdir(save_path, parents=True, exist_ok=True)
        writer = SummaryWriter(save_path)
        self.logger.info(f"INIT::Tensorboard file saved at {save_path}")
        # self.aim_logger["hparams"] = tools.flatten(tools.convert_to_dict(self.cfg))
        return writer

    def allocate_clients_to_rounds(self):
        mapping_dict = defaultdict(list)
        for r in range(self.cfg.federated_settings.comm_round):
            if self.cfg.federated_settings.client_sample < 1.0:
                num_clients = int(
                    self.cfg.federated_settings.client_number
                    * self.cfg.federated_settings.client_sample
                )
                client_list = random.sample(
                    range(self.cfg.federated_settings.client_number), num_clients
                )
            else:
                num_clients = self.cfg.federated_settings.client_number
                client_list = list(range(num_clients))
            mapping_dict[r] = client_list

        self.mapping_dict = mapping_dict

    def init_dataloaders(self):
        match self.cfg.datasets.dataset:
            case "cifar10":
                from src.data_preprocessing.cifar10.data_loader import (
                    get_client_dataloader,
                    get_client_idxes_dict,
                )
            case "cifar100":
                from src.data_preprocessing.cifar100.data_loader import (
                    get_client_dataloader,
                    get_client_idxes_dict,
                )
            case "fmnist":
                from src.data_preprocessing.fmnist.data_loader import (
                    get_client_dataloader,
                    get_client_idxes_dict,
                )
            case "tinyimagenet":
                from src.data_preprocessing.ImageNet.data_loader import (
                    get_client_dataloader,
                    get_client_idxes_dict,
                )
            case _:
                raise ValueError("Unrecognized Dataset!")
        (
            # train_ds,
            # test_ds,
            self.dict_client_idexes,
            self.class_num,
            self.client_infos,
        ) = get_client_idxes_dict(
            self.cfg.datasets.datadir,
            self.cfg.datasets.partition_method,
            self.cfg.datasets.partition_alpha,
            self.cfg.federated_settings.client_number,
        )
        # get_client_dataloader = functools.partial(get_client_dataloader, train_ds, test_ds)

        self.test_dl = get_client_dataloader(
            self.cfg.datasets.datadir,
            self.cfg.datasets.batch_size,
            self.dict_client_idexes,
            client_idx=None,
            train=False,
        )
        self.get_client_dataloader = get_client_dataloader
        # self.logger.info(f"INIT::Data Partation\n{self.dict_client_idexes}")
        with (self.cfg.paths.output_dir / "dict_client_idexes.log").open("w") as f:
            # with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(self.dict_client_idexes, file=f)

        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.client_infos).fillna(0)  # 使用 0 填充缺失值

        plt.figure(
            figsize=(self.cfg.federated_settings.client_number, self.class_num // 2)
        )
        sns.heatmap(
            df, annot=True, cmap="viridis", square=True, fmt="g", annot_kws={"size": 7}
        )

        plt.xlabel("Clients")
        plt.ylabel("Classes")
        plt.title("Data Distribution")

        plt.savefig(
            self.cfg.paths.output_dir / "Data_Distribution.png", bbox_inches="tight"
        )
        plt.show()

        # # 将图像上传到 wandb
        # wandb.log(
        #     {
        #         "Data Distribution": wandb.Image(
        #             str(self.cfg.paths.output_dir / "Data_Distribution.png")
        #         )
        #     }
        # )

    def init_methods(self):
        self.server_dict = {
            "train_data": self.test_dl,
            "test_data": self.test_dl,
            "device": self.device,
            "logger_method": self.logger_method,
        }
        client_dict = {
            "train_data": self.dict_client_idexes,
            "test_data": self.dict_client_idexes,
            "get_dataloader": self.get_client_dataloader,
            "device": torch.device("cuda")
            if self.device.type == "cuda"
            else self.device,
            "client_infos": self.client_infos,
            "logger_method": self.logger_method,
        }

        match self.cfg.method.method_name:
            case "fedavg":
                import src.methods.fedavg as alg
            case "fedprox":
                import src.methods.fedprox as alg
            case "moon":
                import src.methods.moon as alg
            case "fedlora":
                import src.methods.fedlora as alg
            case "feddyn":
                import src.methods.feddyn as alg
            case _:
                raise ValueError(
                    "Invalid --method chosen! Please choose from availible methods."
                )
        self.server = alg.Server.remote(self.server_dict, self.cfg)

        clients = ray.get(
            [
                alg.init_client.remote(client_dict, self.cfg, index)
                for index in range(self.cfg.federated_settings.client_number)
            ]
        )

        self.clients = {index: x for index, x in enumerate(clients)}
        self.train_func = alg.train

    def run_one_round(self, r):
        client_outputs = ray.get(
            [
                self.train_func.remote(self.clients[idx], self.server_outputs, r)
                for idx in self.mapping_dict[r]
            ]
        )
        res_for_log = {"client_results": client_outputs}
        self.server_outputs, server_res = ray.get(
            self.server.run.remote(client_outputs)
        )
        res_for_log.update({"server_results": server_res})
        self.log(round=r + 1, contents=res_for_log)

        return server_res["acc"]

    def log(self, round: int, contents: dict):
        """
        {
            "weights": weights,
            "num_samples": num_samples,
            "client_index": self.client_index,
            "result": dict(**train_res, **val_res),
        }
        """
        # res = contents["client_results"]
        for client_res in contents["client_results"]:
            #  res = client_res["results"]
            for key, value in client_res["result"].items():
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        self.writer.add_scalar(
                            tag=f"client_{client_res['client_index']}/{round}/{key}",
                            scalar_value=item,
                            global_step=index,
                        )
                        # self.aim_logger.track(
                        #     value=item,
                        #     epoch=round,
                        #     step=index,
                        #     name=f"client_{client_res['client_index']}/{round}/{key}",
                        # )
                        # wandb.log(
                        #     {f"client_{client_res['client_index']}/{round}/{key}": item},step=index
                        # )
                else:
                    self.writer.add_scalar(
                        tag=f"client_{client_res['client_index']}/{key}",
                        scalar_value=value,
                        global_step=round,
                    )
                    # self.aim_logger.track(
                    #     value=value,
                    #     epoch=round,
                    #     # step=index,
                    #     name=f"client_{client_res['client_index']}/{key}",
                    # )
                    # wandb.log(
                    #     {f"client_{client_res['client_index']}/{key}": value},
                    #     step=round,
                    # )
        for key, value in contents["server_results"].items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    self.writer.add_scalar(
                        tag=f"server/{round}/{key}",
                        scalar_value=item,
                        global_step=index,
                    )
                    # self.aim_logger.track(
                    #     value=item,
                    #     epoch=round,
                    #     step=index,
                    #     name=f"server/{round}/{key}",
                    # )
                    # wandb.log({f"server/{round}/{key}": item},step=index)
            else:
                self.writer.add_scalar(
                    tag=f"server/{key}",
                    scalar_value=value,
                    global_step=round,
                )
                # self.aim_logger.track(
                #     value=value,
                #     epoch=round,
                #     # step=index,
                #     name=f"server/{key}",
                # )
                # wandb.log(
                #     {f"server/{key}": value},
                #     step=round,
                # )

    def run(self):
        ray.init(
            runtime_env={"env_vars": {"AUTOSCALER_RESOURCE_TYPES_TO_AUTO_ALLOW": "GPU"}}
        )
        self.init_dataloaders()
        self.logger.info("INIT::Data partation finished...")
        self.allocate_clients_to_rounds()
        self.init_methods()
        self.logger.info("INIT::Starting server...")
        self.server_outputs = ray.get(self.server.start.remote())
        time.sleep(5 * (self.cfg.federated_settings.client_number / 100))
        self.logger.info("INIT::Runnging FL...")
        with tqdm.tqdm(range(self.cfg.federated_settings.comm_round)) as t:
            for r in range(self.cfg.federated_settings.comm_round):
                acc = self.run_one_round(r)
                t.set_postfix({"Acc": acc})
                t.set_description(
                    f"""Round: {r+1}/{self.cfg.federated_settings.comm_round}"""
                )
                t.update(1)

        tools.parser_log(self.cfg.paths.output_dir / "server.log")
        self.logger.info("RESULT::Experiment finished...")

        ray.shutdown()
        # wandb.close()
