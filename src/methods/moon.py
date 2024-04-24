"""
Code credit to https://github.com/QinbinLi/MOON
for implementation of thier method, MOON.
"""

import ray
import torch
from src.methods_ray.base import Base_Client, Base_Server
from copy import deepcopy


class Client(Base_Client):
    def __init__(self, client_dict, args, client_index):
        super().__init__(client_dict, args, client_index)
        self.prev_model = deepcopy(self.model)
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def run(self, global_params, round):
        self.model = self.model.to(self.device)
        self.prev_model = deepcopy(self.model).to(self.device)
        super().load_client_state_dict(global_params)
        self.global_model = deepcopy(self.model).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.local_setting.lr,
            momentum=0.9,
            weight_decay=self.args.local_setting.wd,
            nesterov=True,
        )

        self.logger.info(f"***********************{round}***********************")

        weights, train_res = self.train()
        if self.args.local_setting.local_valid:  # and round == last_round:
            self.weight_test = self.get_cdist_test(self.client_index).reshape((1, -1))
            self.acc_dataloader = self.test_dataloader
            val_res = self.test()
        else:
            val_res = {}

        return {
            "weights": weights,
            "num_samples": self.num_samples,
            "client_index": self.client_index,
            "result": dict(**train_res, **val_res),
        }

    def train(self):
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.local_setting.epochs):
            batch_loss = []
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logging.info(x.shape)
                x, target = x.to(self.device), target.to(self.device).long()
                self.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    #####
                    pro1, out = self.model(x)
                    pro2, _ = self.global_model(x)

                    posi = self.cos(pro1, pro2)
                    logits = posi.reshape(-1, 1)

                    pro3, _ = self.prev_model(x)
                    nega = self.cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    logits /= self.hypers.temp
                    labels = torch.zeros(x.size(0)).to(self.device).long()

                    loss2 = self.args.method.hyperparams * self.criterion(logits, labels)

                    loss1 = self.criterion(out, target)
                    loss = loss1 + loss2
                #####
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.logger.info(
                    "(Local Training Epoch: {} \tBatch {}: \tLoss: {:.6f}".format(
                        epoch,
                        batch_idx,
                        sum(epoch_loss) / len(epoch_loss),
                    )
                )
        weights = self.model.cpu().state_dict()
        return weights, {"train_loss_epoch": epoch_loss}

    def test(self):
        self.model.eval()

        preds = None
        labels = None
        acc = None
        losses = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                _, pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                losses.append(loss)
                if preds is None:
                    preds = predicted.cpu()
                    labels = target.cpu()
                else:
                    preds = torch.concat((preds, predicted.cpu()), dim=0)
                    labels = torch.concat((labels, target.cpu()), dim=0)
        for c in range(self.num_classes):
            temp_acc = (
                (
                    ((preds == labels) * (labels == c)).float()
                    / (max((labels == c).sum(), 1))
                )
                .sum()
                .cpu()
            )
            if acc is None:
                acc = temp_acc.reshape((1, -1))
            else:
                acc = torch.concat((acc, temp_acc.reshape((1, -1))), dim=0)
        # print(acc.device,self.weight_test.device)
        weighted_acc = acc.reshape((1, -1)).mean()
        self.logger.info(
            "************* Client {} Acc = {:.2f} **************".format(
                self.client_index, weighted_acc.item()
            )
        )
        return {
            "val_loss": sum(losses) / len(losses),
            "val_weighted_acc": weighted_acc.item(),
            "val_acc": (
                (preds == labels).float().sum() / (labels).float().sum()
            ).item(),
        }

    def load_client_state_dict(self, server_state_dict):
        super().load_client_state_dict(server_state_dict)

    def get_cdist(self):
        return super().get_cdist()


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

    def test(self):
        return super().test()

    def compute_grad_norm(self):
        return super().compute_grad_norm()

    def test_inner(self, data):
        self.model = self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        # test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(data):
                x = x.to(self.device)
                target = target.to(self.device)

                _, pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item()# * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
        return acc, test_sample_number


@ray.remote(num_gpus=0.09)
def train(client: Client, gloabl_params, round):
    return client.run(gloabl_params, round)


@ray.remote(num_cpus=1)
def init_client(client_dict, args, client_index):
    return Client(client_dict, args, client_index)
