import numpy as np
import matplotlib.pyplot as plt
import os
import json

adaptive_filter = lambda x: '_A_' in x
non_adaptive_filter = lambda x: '_A_' not in x
model_filter = lambda x, model: model in x


class RunAnalyser:
    def __init__(self, runs, test_path):
        self.runs = runs
        self.mses = self.load_mses(runs, test_path)
        self.meta_data = self.load_meta_data(runs, test_path)
        self.test_path = test_path

    def load_mses(self, runs, test_path):
        mses = dict()
        for model in runs:
            mses[model] = np.load(os.path.join(test_path, model, 'test_mse.npz'))['mse']
        return mses
    
    def load_meta_data(self, runs, test_path):
        meta_data = dict()
        for model in runs:
            meta_data[model] = json.load(open(os.path.join(test_path, model, 'test_meta.json')))
        return meta_data

    def check_start_end(self, start, end):
        if end is None:
            end = max([len(model_mse) for model_mse in self.mses.values()])
        assert start >= 0, "Start index must be non-negative"
        assert end > start, "End index must be greater than start index"
        assert end <= max([len(model_mse) for model_mse in self.mses.values()]), "End index exceeds maximum length of MSE data"
        return start, end

    def plot_mses(self, filter=lambda _: True, test_name='Test MSE', start=0, end=None, figsize=(10, 5)):
        plt.figure(figsize=figsize)

        if end is None:
            end = max([len(model_mse) for model_mse in self.mses.values()])

        start, end = self.check_start_end(start, end)

        for model, model_mse in self.get_mses(filter).items():
            # Plot the MSE for each model
            plt.semilogy(model_mse[start:min(end, len(model_mse))], label=model)

        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title(test_name)
        plt.legend()
        plt.show()
    
    def get_mses(self, filter=lambda _: True, start=0, end=None):
        start, end = self.check_start_end(start, end)
        mses = dict()

        for model, model_mse in self.mses.items():
            if not filter(model):
                continue
            # Get the MSE for each model
            mses[model] = model_mse

        return mses

    def get_late_variances(self, filter=lambda _: True, stability_pct=0.1):
        late_variances = dict()

        for model, model_mse in self.mses.items():
            if not filter(model):
                continue
            # stability check (late stage variance)
            n = len(model_mse)
            stability_start = int(n * (1 - stability_pct))
            cut = model_mse[stability_start:]
            late_variances[model] = np.std(cut) / np.mean(cut)

        return late_variances

    def get_mean_absolute_change(self, filter=lambda _: True, start=0, end=None):
        mean_absolute_changes = dict()

        start, end = self.check_start_end(start, end)

        for model, model_mse in self.mses.items():
            if not filter(model):
                continue
            cut = model_mse[start:min(end, len(model_mse))]
            # mean absolute change
            mean_absolute_changes[model] = np.mean(np.abs(np.diff(cut)))

        return mean_absolute_changes

    def get_flops(self, filter=lambda _: True):
        flops = dict()

        for model, meta in self.meta_data.items():
            if not filter(model):
                continue
            # Get the FLOPs from the meta data
            flops[model] = meta['flops']

        return flops

    def get_num_params(self, filter=lambda _: True):
        num_params = dict()

        for model, meta in self.meta_data.items():
            if not filter(model):
                continue
            # Get the number of parameters from the meta data
            num_params[model] = meta['param_count']

        return num_params
    
    def get_training_time(self, filter=lambda _: True):
        training_time = dict()

        for model, meta in self.meta_data.items():
            if not filter(model):
                continue
            # Get the training time from the meta data
            training_time[model] = meta['training_time']

        return training_time
    
    def plot_mse_over_params(self, filter=lambda _: True, title='MSE vs Params', xlabel='num_params', ylabel='mse loss', noshow=False, legend=True):
        if not noshow:
            plt.figure(figsize=(7, 7))

        num_params = self.get_num_params(filter)
        mses = self.get_mses(filter)

        slice_mse = 0.1
        start, end = self.check_start_end(0, None)
        start = int(end * (1 - slice_mse))

        for model in self.runs:
            if not filter(model):
                continue
            
            mse = np.mean(mses[model][start:end])
            params = num_params[model]
            plt.scatter(params, mse, label=model, cmap='viridis')

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend()
        if not noshow:
            plt.show()

    def plot_mse_over_flops(self, filter=lambda _: True, title='MSE vs FLOPs', xlabel='flops', ylabel='mse loss', noshow=False, legend=True):
        if not noshow:
            plt.figure(figsize=(7, 7))

        flops = self.get_flops(filter)
        mses = self.get_mses(filter)

        slice_mse = 0.1
        start, end = self.check_start_end(0, None)
        start = int(end * (1 - slice_mse))

        for model in self.runs:
            if not filter(model):
                continue
            
            plt.scatter(flops[model], np.mean(mses[model][start:end]), label=model)

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend()
        if not noshow:
            plt.show()
    
    def plot_flops_over_params(self, filter=lambda _: True, title='FLOPs vs Params', xlabel='num_params', ylabel='flops', noshow=False):
        if not noshow:
            plt.figure(figsize=(10, 5))

        flops = self.get_flops(filter)
        num_params = self.get_num_params(filter)

        for model in self.runs:
            if not filter(model):
                continue
            
            plt.scatter(num_params[model], flops[model], label=model)

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        if not noshow:
            plt.show()

    def plot_prediction_over_actual_1D(self, models, colors=None, epoch=10000, figsize=(10, 5), exact_color='gainsboro'):
        
        pred_file = f'test_{epoch}.npy'
        exact_file = 'test_exact.npy'
        if colors is None:
            colors=['black'] * len(models)

        for model, color in zip(models, colors):
            pred_path = os.path.join(self.test_path, model, pred_file)
            exact_path = os.path.join(self.test_path, model, exact_file)
            if not os.path.exists(pred_path) or not os.path.exists(exact_path):
                continue
            pred = np.load(pred_path)
            exact = np.load(exact_path)
            if pred.shape != exact.shape:
                print(f"Shape mismatch for model {model}: {pred.shape} vs {exact.shape}")
                continue

            # Plot the prediction vs actual
            plt.figure(figsize=figsize)
            plt.plot(exact, label='actual', linestyle='-', lw=4, color=exact_color)
            plt.plot(pred, label=f'{model} prediction', linestyle='--', color=color, linewidth=1)
            plt.title(f'Prediction vs Actual for {model}')
            plt.legend()
            plt.show()