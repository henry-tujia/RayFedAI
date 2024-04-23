"""
Author: henry tanhao0606@outlook.com
Date: 2024-04-02 06:53:19
LastEditors: henry tanhao0606@outlook.com
LastEditTime: 2024-04-02 06:53:19
FilePath: /Multi-FL-Training-main/src/methods/fedlora.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import ray
import torch
from src.methods_ray.base import Base_Client, Base_Server
import torch.nn as nn
import copy


class Client(Base_Client):
    def __init__(self, client_dict, args, client_index):
        super().__init__(client_dict, args, client_index)
        self.client_grad = self.build_grad_dict(self.model)

    def build_grad_dict(self, model):
        grad_dict = {}
        for key, params in model.state_dict().items():
            grad_dict[key] = torch.zeros_like(params)
        return grad_dict

    def load_client_state_dict(self, server_state_dict):
        super().load_client_state_dict(server_state_dict)

    def train(self):
        glo_model = copy.deepcopy(self.model)
        glo_model.eval()
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.local_setting.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                # origin_model = copy.deepcopy(self.model)
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)
                    # FedDyn Loss
                reg_loss = 0.0
                cnt = 0.0
                for name, param in self.model.named_parameters():
                    term1 = (
                        param.cpu()
                        * (self.client_grad[name] - glo_model.cpu().state_dict()[name])
                    ).sum()
                    term2 = (param.cpu() * param.cpu()).sum()

                    reg_loss += self.args.method.hyperparams.mu * (term1 + term2)
                    cnt += 1.0

                loss += reg_loss / cnt
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.method.hyperparams.max_grad_norm
                )
                self.optimizer.step()
                batch_loss.append(loss.item())
                # break
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.logger.info(
                    "(Local Training Epoch: {} \tBatch {}: \tLoss: {:.6f}".format(
                        epoch,
                        batch_idx,
                        sum(epoch_loss) / len(epoch_loss),
                    )
                )
        for name, param in self.model.named_parameters():
            self.client_grad[name] += (
                self.model.cpu().state_dict()[name] - glo_model.cpu().state_dict()[name]
            )
        weights = self.model.cpu().state_dict()
        return weights, {"train_loss_epoch": epoch_loss}

    def run(self, received_info, round):
        client_results = super().run(received_info, round)

        client_results.update(
            {
                "client_grad": self.client_grad,
            }
        )

        return client_results

    def get_cdist(self):
        return super().get_cdist()

    def test(self):
        return super().test()


@ray.remote(num_gpus=0.05)
class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        # global grad dict
        self.glo_grad = self.build_grad_dict(self.model)

    def run(self, received_info):
        return super().run(received_info)

    def start(self):
        return super().start()

    def operations(self, client_info):
        # 按client_index排序
        client_info.sort(key=lambda tup: tup["client_index"])

        client_model_weights = [c["weights"] for c in client_info]
        client_samples = [c["num_samples"] for c in client_info]
        total_samples = sum(client_samples)

        # 计算每个客户端的权重
        client_weights = [s / total_samples for s in client_samples]

        mean_state_dict = {}

        for name, param in self.model.cpu().state_dict().items():
            vs = []
            for client_model_weight in client_model_weights:
                vs.append(client_model_weight[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()

            alpha = self.args.federated_settings.client_sample
            mean_state_dict[name] = alpha * mean_value + (1.0 - alpha) * param

        # global_model.load_state_dict(mean_state_dict, strict=False)
        self.model.load_state_dict(mean_state_dict, strict=False)

        return self.model.cpu().state_dict()

    def build_grad_dict(self, model):
        grad_dict = {}
        for key, params in model.state_dict().items():
            grad_dict[key] = torch.zeros_like(params).cpu()
        return grad_dict

    def test_inner(self, data):
        return super().test_inner(data)

    def test(self):
        return super().test()

    def compute_grad_norm(self):
        return super().compute_grad_norm()


@ray.remote(num_gpus=0.09)
def train(client: Client, gloabl_params, round):
    return client.run(gloabl_params, round)


@ray.remote(num_cpus=1)
def init_client(client_dict, args, client_index):
    return Client(client_dict, args, client_index)
