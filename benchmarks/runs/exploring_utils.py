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

    def plot_mses(self, filter=lambda _: True, test_name='Test MSE', start=0, end=None, figsize=(10, 5), noshow=False):
        if not noshow:
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
        if not noshow:
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
            flops[model] = meta['flops'][0]

        return flops
    
    def get_training_time(self, filter=lambda _: True):
        flops = dict()

        for model, meta in self.meta_data.items():
            if not filter(model):
                continue
            # Get the FLOPs from the meta data
            flops[model] = meta['training_time']

        return flops

    def get_num_params(self, filter=lambda _: True):
        num_params = dict()

        for model, meta in self.meta_data.items():
            if not filter(model):
                continue
            # Get the number of parameters from the meta data
            num_params[model] = meta['param_count'][0]

        return num_params
    
    def get_training_time(self, filter=lambda _: True):
        training_time = dict()

        for model, meta in self.meta_data.items():
            if not filter(model):
                continue
            # Get the training time from the meta data
            training_time[model] = meta['training_time']

        return training_time
    
    def get_training_time_from_list(self, l=[]):
        training_time = dict()

        for model in l:
            if model not in self.meta_data:
                continue
            # Get the training time from the meta data
            training_time[model] = self.meta_data[model]['training_time']
        return training_time
    
    def plot_log_log(self, models, xs, ys, xlabel, ylabel, title, legend=False, noshow=False):
        if not noshow:
            plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap('tab20', len(models))  # Use a colormap for consistent colors

        for i, model in enumerate(models):
            # Scatter plot with distinct colors and larger markers
            plt.scatter(xs[model], ys[model], label=model, s=100, alpha=0.7, color=colors(i))

            # Annotate the single point
            plt.annotate(model, (xs[model], ys[model]), fontsize=8, alpha=0.8)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)

        if legend:
            # Place legend outside the plot area with multiple columns
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize=10, title="Models")

        if not noshow:
            plt.show()

        
    def plot_mse_over_params(self, filter=lambda _: True, title='MSE vs Params', xlabel='num_params', ylabel='mse loss', noshow=False, legend=False):
        if not noshow:
            plt.figure(figsize=(7, 7))

        num_params = self.get_num_params(filter)
        mses = self.get_mses(filter)

        slice_mse = 0.1
        # start, end = self.check_start_end(0, None)
        # start = int(end * (1 - slice_mse))

        models = []
        ys = {}
        for model in self.runs:
            if not filter(model):
                continue
            models.append(model)
            end = len(mses[model])
            start = int(end * (1 - slice_mse))
            ys[model] = np.mean(mses[model][start:end])

        self.plot_log_log(models, num_params, ys, xlabel, ylabel, title, legend, noshow)


    def plot_mse_over_flops(self, filter=lambda _: True, title='MSE vs FLOP', xlabel='flops', ylabel='mse loss', noshow=False, legend=False):
        if not noshow:
            plt.figure(figsize=(7, 7))

        flops = self.get_flops(filter)
        mses = self.get_mses(filter)

        slice_mse = 0.1
        # start, end = self.check_start_end(0, None)
        # start = int(end * (1 - slice_mse))

        models = []
        ys = {}
        for model in self.runs:
            if not filter(model):
                continue
            models.append(model)
            end = len(mses[model])
            start = int(end * (1 - slice_mse))
            ys[model] = np.mean(mses[model][start:end])

        self.plot_log_log(models, flops, ys, xlabel, ylabel, title, legend, noshow)

    
    def plot_flops_over_params(self, filter=lambda _: True, title='FLOP vs Params', xlabel='num_params', ylabel='flops', noshow=False):
        if not noshow:
            plt.figure(figsize=(10, 5))

        flops = self.get_flops(filter)
        num_params = self.get_num_params(filter)

        models = []
        for model in self.runs:
            if not filter(model):
                continue
            models.append(model)
        
        self.plot_log_log(models, num_params, flops, xlabel, ylabel, title, False, noshow)


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
    
    def plot_training_times(self, filter=lambda _: True, title='Training Time', xlabel='Model', ylabel='Time (s)', figsize=(10, 5), noshow=False):
        if not noshow:
            plt.figure(figsize=figsize)
        training_times = self.get_training_time(filter)
        models = list(training_times.keys())
        times = list(training_times.values())

        def get_color(model):
            if 'FCN' in model:
                return 'blue'
            elif 'CKAN' in model:
                return 'orange'
            elif 'LKAN' in model:
                return 'green'
            else:
                return 'gray'

        plt.bar(models, times, color=[get_color(model) for model in models], alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if not noshow:
            plt.show()

    def plot_training_times_from_list(self, l=[], title='Training Time', xlabel='Model', ylabel='Time (s)', figsize=(10, 5), noshow=False):
        if not noshow:
            plt.figure(figsize=figsize)
        training_times = self.get_training_time_from_list(l)
        models = list(training_times.keys())
        times = list(training_times.values())

        def get_color(model):
            if 'FCN' in model:
                return 'blue'
            elif 'Optimized' in model:
                return 'orangered'
            elif 'CKAN' in model:
                return 'orange'
            elif 'LKAN' in model:
                return 'green'
            else:
                return 'gray'

        plt.bar(models, times, color=[get_color(model) for model in models], alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if not noshow:
            plt.show()

    def plot_flops_over_params(self, filter=lambda _: True, title='FLOP vs Params', xlabel='num_params', ylabel='flops', noshow=False):
        if not noshow:
            plt.figure(figsize=(10, 5))

        flops = self.get_flops(filter)
        num_params = self.get_num_params(filter)

        models = []
        for model in self.runs:
            if not filter(model):
                continue
            models.append(model)
        
        self.plot_log_log(models, num_params, flops, xlabel, ylabel, title, False, noshow)