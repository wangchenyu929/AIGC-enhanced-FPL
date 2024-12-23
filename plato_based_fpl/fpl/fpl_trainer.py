
import time
import logging
import torch
import math
import multiprocessing as mp

from plato.config import Config
from plato.trainers import basic
from torch.utils.data import DataLoader
from itertools import zip_longest
# from plato.datasources.aug_cifar10 import aug
# from torchvision import transforms
# from torchvision.transforms import ToPILImage
# import numpy as np


class Trainer(basic.Trainer):
    

     ## add two avgs of this func: aigc_trainset and aigc_sampler
    def train(self, trainset, aigc_trainset, aug_trainset, sampler, aigc_sampler, aug_sampler) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        kwargs (optional): Additional keyword arguments.

        Returns:
        float: Elapsed time during training.
        """
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]

        # Set the start time of training in absolute time
        self.training_start_time = time.time()

        if "max_concurrency" in config:
            tic = time.perf_counter()

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            ## refine the args of train_process
            train_proc = mp.Process(
                target=self.train_process,
                args=(config, trainset, aigc_trainset, aug_trainset, sampler, aigc_sampler, aug_sampler),
            )
            train_proc.start()
            train_proc.join()

            model_name = Config().trainer.model_name
            filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.pth"

            try:
                self.load_model(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Training on client {self.client_id} failed."
                ) from error

            toc = time.perf_counter()
            self.pause_training()
        else:
            tic = time.perf_counter()
            self.train_process(config, trainset, aigc_trainset, aug_trainset, sampler, aigc_sampler, aug_sampler)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time

    ## add two avgs of this func: aigc_trainset and aigc_sampler
    
    # def get_aug(self, x):
    #     preprocess = transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.Normalize(
    #                         [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ])
    #     return aug(x, preprocess)
  
    def train_process(self, config, trainset, aigc_trainset, aug_trainset,  sampler, aigc_sampler, aug_sampler):
        """
        The main training loop in a federated learning workload, run in a
        separate process with a new CUDA context, so that CUDA memory can be
        released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: The sampler that extracts a partition for this client.
        kwargs (optional): Additional keyword arguments.
        """
        try:
            ## add two args: aigc_sampler and aigc_sampler.get()
            # self.train_model(config, trainset, aigc_trainset, sampler.get(), aigc_sampler.get())
            self.train_model(config, trainset, aigc_trainset, aug_trainset, sampler, aigc_sampler, aug_sampler)
        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if "max_concurrency" in config:
            self.model.cpu()
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

    ## add two avgs of this func: aigc_trainset and aigc_sampler
    def train_model(self, config, trainset, aigc_trainset, aug_trainset, sampler, aigc_sampler, aug_sampler):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler.get()
        self.aug_sampler = aug_sampler.get()
        
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)
        
        ## add two avgs of this func: aigc_trainset and aigc_sampler
        self.train_loader,self.aigc_train_loader,self.aug_train_loader = self.get_train_loader(batch_size, trainset, aigc_trainset, aug_trainset, sampler, aigc_sampler, aug_sampler)


        # Initializing the loss criterion
        self._loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        self.optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, self.optimizer)
        self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)
            
            all = 0
            num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            if self.aigc_train_loader and self.aug_train_loader:
                for batch_id, (data1,data2,data3) in enumerate(zip_longest(self.train_loader,self.aigc_train_loader,self.aug_train_loader)):
                    
                    tar_len = len(data2[0])
                    
                    if (len(data3[0]) < tar_len):
                        tar_len = tar_len - len(data3[0])
                        data3[0] = torch.cat((data3[0], data3[0][:tar_len, :, :, :] ),dim=0)
                        data3[1] = torch.cat((data3[1], data3[1][:tar_len] ),dim=0)
                    else:
                        data3[0] = data3[0][:tar_len, :, :, :] 
                        data3[1] = data3[1][:tar_len] 
                    
                    if data1 and data2 and data3:
                        examples = torch.cat((data1[0],data2[0],data3[0]),dim=0)
                        labels = torch.cat((data1[1],data2[1],data3[1]),dim=0)
                    elif(data1 and not data2):
                        examples = data1[0]
                        labels = data1[1]
                    elif(not data1 and data2):
                        examples = data2[0]
                        labels = data2[1]
                        
                    for i in labels:
                        num[i] += 1
                        all += 1

                    self.train_step_start(config, batch=batch_id)
                    self.callback_handler.call_event(
                        "on_train_step_start", self, config, batch=batch_id
                    )

                    examples, labels = examples.to(self.device), labels.to(self.device)
                    
                    print('len of original data:', len(data1[0]), 'len of data after aug:', len(examples))

                    loss = self.perform_forward_and_backward_passes(
                        config, examples, labels
                    )

                    self.train_step_end(config, batch=batch_id, loss=loss)
                    self.callback_handler.call_event(
                        "on_train_step_end", self, config, batch=batch_id, loss=loss
                    )
            # elif not self.aigc_train_loader and self.aug_train_loader:
            #     for batch_id, (data1,data2) in enumerate(zip_longest(self.train_loader,self.aug_train_loader)):
                    
            #         class_num = []
            #         class_pos = []
            #         aug_data = [[], []]
            #         for i in range(10):
            #             class_num.append(0)
            #             class_pos.append([])

            #         for i in range(len(data1[1])):
            #             class_num[data1[1][i]] += 1
            #             class_pos[data1[1][i]].append(i)
                        
            #         max_num = max(class_num)
                    
            #         for i in range(10):
            #             num_i = class_num[i]
            #             if num_i and num_i < max_num:
            #                 a = max_num // num_i
            #                 for _ in range(a):
            #                     for t in range(len(class_pos[i])):
            #                         aug_data[0].append(self.get_aug(ToPILImage()(data2[0][class_pos[i][t]])))
            #                         aug_data[1].append(i)
                                    
            #         new_tensor = torch.stack(aug_data[0])
            #         new_tar = torch.tensor(aug_data[1])
                    
            #         aug_d = [new_tensor, new_tar]
                    
            #         examples = torch.cat((data1[0],aug_d[0]),dim=0)
            #         labels = torch.cat((data1[1],aug_d[1]),dim=0)
                    
            #         print("batch len after aug is ", len(examples))
                    
            #         # if data1 and data2:
            #         #     examples = torch.cat((data1[0],data2[0]),dim=0)
            #         #     labels = torch.cat((data1[1],data2[1]),dim=0)
            #         #     print(labels)
            #         # elif(data1 and not data2):
            #         #     examples = data1[0]
            #         #     labels = data1[1]
            #         # elif(not data1 and data2):
            #         #     examples = data2[0]
            #         #     labels = data2[1]

            #         self.train_step_start(config, batch=batch_id)
            #         self.callback_handler.call_event(
            #             "on_train_step_start", self, config, batch=batch_id
            #         )

            #         examples, labels = examples.to(self.device), labels.to(self.device)

            #         loss = self.perform_forward_and_backward_passes(
            #             config, examples, labels
            #         )

            #         self.train_step_end(config, batch=batch_id, loss=loss)
            #         self.callback_handler.call_event(
            #             "on_train_step_end", self, config, batch=batch_id, loss=loss
            #         )
                
            else:
                for batch_id, (examples, labels) in enumerate(self.train_loader):
                    # print(f"client #{self.client_id}:len of examples:{len(labels)} in batch:{batch_id}")
                    
                    self.train_step_start(config, batch=batch_id)
                    self.callback_handler.call_event(
                        "on_train_step_start", self, config, batch=batch_id
                    )

                    examples, labels = examples.to(self.device), labels.to(self.device)

                    loss = self.perform_forward_and_backward_passes(
                        config, examples, labels
                    )

                    self.train_step_end(config, batch=batch_id, loss=loss)
                    self.callback_handler.call_event(
                        "on_train_step_end", self, config, batch=batch_id, loss=loss
                    )
            
            self.lr_scheduler_step()
            
            print("class 1:", num[0], "class 2:", num[1], all, num[0] / all, num[1] / all)

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    ## add two avgs of this func: aigc_trainset and aigc_sampler
    def get_train_loader(self, batch_size, trainset, aigc_trainset, aug_trainset, sampler, aigc_sampler, aug_sampler):
        """
        Creates an instance of the trainloader.

        Arguments:
        batch_size: the batch size.
        trainset: the training dataset.
        aigc_trainset: the aigc dataset.
        sampler: the sampler for the trainloader to use.
        aigc_sampler: how aigc data is allocated on the client.
        """
        if aigc_sampler:
            # calculate the batch size of local dataloader and aigc data loader
            local_samples_number = Config().data.partition_size
            aigc_samples_number = aigc_sampler.num_samples()
            if local_samples_number>aigc_samples_number:
                local_batch_size = batch_size
                batch_round = math.ceil(local_samples_number/batch_size)
                aigc_batch_size = int(math.ceil(aigc_samples_number/batch_round))
            else:
                aigc_batch_size = batch_size
                batch_round = math.ceil(aigc_samples_number/batch_size)
                local_batch_size = int(math.ceil(local_samples_number/batch_round))

            local_dataloader = DataLoader(
                dataset=trainset, shuffle=False, batch_size=local_batch_size, sampler=sampler.get()
            )

            aigc_dataloader = DataLoader(
                dataset=aigc_trainset, shuffle=False, batch_size=aigc_batch_size, sampler=aigc_sampler.get()
            )
            
            aug_dataloader = DataLoader(
                dataset=aug_trainset, shuffle=False, batch_size=local_batch_size, sampler=sampler.get()
            )


        else:
            # there is no need to 
            local_dataloader = DataLoader(
                dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler.get()
            )
            aug_dataloader = DataLoader(
                dataset=aug_trainset, shuffle=False, batch_size=batch_size, sampler=aug_sampler.get()
            )
            aigc_dataloader = None

        return local_dataloader,aigc_dataloader,aug_dataloader


    def test_process(self, config, testset, sampler=None, **kwargs):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.
        """
        self.model.to(self.device)
        self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            if sampler is None:
                accuracy = self.test_model(config, testset, **kwargs)
            else:
                accuracy = self.test_model(config, testset, sampler.get(), **kwargs)

        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        if "max_concurrency" in config:
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy