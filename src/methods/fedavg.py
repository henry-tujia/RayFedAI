import ray
from src.methods_ray.base import Base_Client, Base_Server


class Client(Base_Client):
    def __init__(self, client_dict, args, client_index):
        super().__init__(client_dict, args, client_index)

    def load_client_state_dict(self, server_state_dict):
        super().load_client_state_dict(server_state_dict)

    def get_cdist(self):
        return super().get_cdist()

    def run(self, global_params, round):
        return super().run(global_params, round)

    def train(self):
        return super().train()

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
