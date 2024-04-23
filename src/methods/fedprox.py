"""
Code credit to https://github.com/QinbinLi/MOON
for thier implementation of FedProx.
"""

import torch
from src.methods_ray.base import Base_Client, Base_Server
import copy
import ray


class Client(Base_Client):
    def __init__(self, client_dict, args, client_index):
        super().__init__(client_dict, args, client_index)

    def train(self):
        self.model.to(self.device)
        global_weight_collector = copy.deepcopy(list(self.model.parameters()))
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.local_setting.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)
                ############
                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += (self.args.method.hyperparams.mu / 2) * torch.norm(
                        (param - global_weight_collector[param_index])
                    ) ** 2
                loss = loss + fed_prox_reg
                ########
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.logger.info(
                    "(Local Training Epoch: {} \tLoss: {:.6f}".format(
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                    )
                )
        weights = self.model.cpu().state_dict()
        return weights, {"train_loss_epoch": epoch_loss}

    def load_client_state_dict(self, server_state_dict):
        super().load_client_state_dict(server_state_dict)

    def get_cdist(self):
        return super().get_cdist()

    def run(self, global_params, round):
        return super().run(global_params, round)

    def test(self):
        return super().test()


@ray.remote(num_gpus=0.05)
class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)

    def run(self, received_info):
        return super().run(received_info)

    def start(self):
        return super().start()

    def operations(self, client_info):
        return super().operations(client_info)

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
