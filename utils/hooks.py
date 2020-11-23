import threading

import torch
import numpy as np
class TBHook(object):
    def __init__(self, model,writer, start_iter=0, n_parallel_devices=0, log_interval=200):
        self.hook = model.register_forward_hook(self.hook_fn)
        self.writer = writer
        self.log_interval = log_interval
        self.n_parallel_devices = max(1,n_parallel_devices)
        self.lock = threading.Lock()

        self.iteration = start_iter
        self.current_train_summaries = {}
        self.current_eval_summaries = {}
        self.device_iterator = 0
        self.eval_iteration = 0

    def flush_eval_summaries(self):
        self.write_to_tensorboard({k: v / self.eval_iteration if np.isscalar(v) else v for k, v in self.current_eval_summaries.items()})

    def reset_eval_summaries(self):
        self.current_eval_summaries = {}
        self.eval_iteration = 0

    def hook_fn(self, module, input,output):
        if module.training:
            if len(self.current_eval_summaries) > 0:
                # We flush the eval summaries
                self.flush_eval_summaries()
                self.reset_eval_summaries()

            if self.iteration % self.log_interval == 0 and hasattr(module,"get_tb_summaries"):
                with torch.no_grad():
                    tb_summaries = module.get_tb_summaries()
                with self.lock:
                    self.device_iterator += 1
                    self.update_summaries(current_summaries=self.current_train_summaries, new_summaries=tb_summaries, scalar_averaging_facvtor=1./self.n_parallel_devices)
                    if self.device_iterator == self.n_parallel_devices:
                        self.write_to_tensorboard(self.current_train_summaries)
                        self.current_train_summaries = {}
                        self.device_iterator = 0
                        self.iteration += 1

            else:
                with self.lock:
                    self.device_iterator += 1
                    if self.device_iterator == self.n_parallel_devices:
                        self.device_iterator = 0
                        self.iteration += 1
        else:
            if hasattr(module, "get_tb_summaries"):
                tb_summaries = module.get_tb_summaries()
                self.update_summaries(current_summaries=self.current_eval_summaries, new_summaries=tb_summaries)
                self.eval_iteration += 1


    def close(self):
        self.hook.remove()

    def update_summaries(self, current_summaries, new_summaries, scalar_averaging_facvtor=1.0):
        # For tensors, we concatenate along dim 0
        # For scalars, we perform averaging
        # current_summaries = self.current_train_summaries
        for k,v in new_summaries.items():
            if v.squeeze().size() == torch.Size():
                if k in current_summaries:
                    current_summaries[k] += v.item()*scalar_averaging_facvtor
                else:
                    current_summaries[k] = v.item()*scalar_averaging_facvtor
            else:
                if k in current_summaries:
                    current_summaries[k] = np.concatenate((current_summaries[k], v.cpu().detach().numpy()), 0)
                else:
                    current_summaries[k] = v.cpu().detach().numpy()

        # self.current_train_summaries = current_summaries



    def write_to_tensorboard(self, summaries):
        # summaries = self.current_train_summaries
        for k, v in summaries.items():
            if np.isscalar(v):
                self.writer.add_scalar(k, v, self.iteration)
            else:
                self.writer.add_histogram(k, v, self.iteration)
        #self.current_train_summaries = {}
