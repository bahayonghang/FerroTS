"""Observe the pinned official train method without editing its source checkout."""
import inspect
import json
import textwrap
import time
from pathlib import Path
import torch


def synchronize(model):
    if model.args.use_gpu:
        torch.cuda.synchronize(model.device)


def begin(model, epoch, batches):
    synchronize(model)
    if model.args.use_gpu:
        torch.cuda.reset_peak_memory_stats(model.device)
    model._p2_observation = dict(epoch=epoch + 1, training_windows=batches * model.args.batch_size,
        steady_windows=max(0, batches - 1) * model.args.batch_size,
        steady_definition="exclude first training batch of each epoch", begin=time.perf_counter())


def first_batch(model):
    synchronize(model)
    model._p2_observation["steady_begin"] = time.perf_counter()


def end_train(model):
    synchronize(model)
    now = time.perf_counter()
    row = model._p2_observation
    row["train_seconds"] = now - row.pop("begin")
    row["steady_seconds"] = now - row.pop("steady_begin", now)
    row["steady_samples_per_second"] = row["steady_windows"] / row["steady_seconds"] if row["steady_seconds"] > 0 else None


def validation(model, name, data, loader, criterion):
    synchronize(model)
    start = time.perf_counter()
    result = model.vali(data, loader, criterion)
    synchronize(model)
    model._p2_observation[name + "_seconds"] = time.perf_counter() - start
    model._p2_observation[name + "_loss"] = float(result)
    return result


def checkpoint(model, early, loss, path, training_loss, optimizer):
    synchronize(model)
    start = time.perf_counter()
    early(loss, model.model, path)
    synchronize(model)
    row = model._p2_observation
    row.update(checkpoint_seconds=time.perf_counter()-start,
               training_loss=float(training_loss), learning_rate=optimizer.param_groups[0]["lr"])
    if model.args.use_gpu:
        row.update(allocator_max_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
                   allocator_max_reserved_bytes=torch.cuda.max_memory_reserved(model.device))
    with Path("performance.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, allow_nan=False) + "\n")


def instrument(source):
    replacements = {
        "epoch_time = time.time()": "_p2_begin(self, epoch, len(train_loader))\n            epoch_time = time.time()",
        "model_optim.step()": "model_optim.step()\n                    if i == 0:\n                        _p2_first_batch(self)",
        'print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))':
            '_p2_end_train(self)\n            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))',
        "self.vali(vali_data, vali_loader, criterion)": "_p2_validation(self, 'validation', vali_data, vali_loader, criterion)",
        "self.vali(test_data, test_loader, criterion)": "_p2_validation(self, 'test', test_data, test_loader, criterion)",
        "early_stopping(vali_loss, self.model, path)": "_p2_checkpoint(self, early_stopping, vali_loss, path, train_loss, model_optim)",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"official instrumentation anchor changed: {old}")
        source = source.replace(old, new)
    return source


def install():
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
    method = Exp_Long_Term_Forecast.train
    # Replace only the method in this process; pinned upstream files stay unchanged.
    source = instrument(inspect.getsource(method))
    namespace = dict(method.__globals__, _p2_begin=begin, _p2_first_batch=first_batch,
        _p2_end_train=end_train, _p2_validation=validation, _p2_checkpoint=checkpoint)
    exec(compile(textwrap.dedent(source), "<p2-performance-observer>", "exec"), namespace)
    Exp_Long_Term_Forecast.train = namespace["train"]
