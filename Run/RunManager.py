import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
import pandas as pd
from collections import OrderedDict
import json


class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        # Add graph to tensorboard
        images, _ = next(iter(loader))
        grid = torchvision.utils.make_grid(images[:10])
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images.to(next(network.parameters()).device))

    def end_run(self):
        """ Cleanup after a run """
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        """ Compute and log epoch duration, loss, and accuracy; update run data """
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        if len(self.loader.dataset) > 0:
            loss = self.epoch_loss / len(self.loader.dataset)
            accuracy = self.epoch_num_correct / len(self.loader.dataset)
        else:
            loss, accuracy = 0, 0

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            if param.grad is not None:
                self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict([
            ("run", self.run_count),
            ("epoch", self.epoch_count),
            ("loss", loss),
            ("accuracy", accuracy),
            ("epoch duration", epoch_duration),
            ("run duration", run_duration)
        ])

        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame(self.run_data)
        print(df.tail(1))  # Print the latest epoch data

    def save(self, fileName):
        df = pd.DataFrame(self.run_data)
        df.to_csv(f'{fileName}.csv', index=False)

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
