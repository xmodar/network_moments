import os
import torch
import importlib
import torchvision  # pylint: disable=W0611
from .vgg import VGG16
from .lenet import LeNet
from shutil import rmtree
from .alexnet import AlexNet
from ..utils import verbosify
from argparse import Namespace


__all__ = ['Trainer', 'ClassifierTrainer', 'TransformedDataset']


class TransformedDataset(torch.utils.data.Dataset):
    @classmethod
    def of(cls, dataset, transform=None, target_transform=None):
        if None is transform is target_transform:
            return dataset
        return cls(
            dataset, transform=transform, target_transform=target_transform)

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data, target = self.dataset[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return len(self.dataset)


class Trainer:
    ignore_zero_loss_coefficients = True

    @classmethod
    def loss(cls, model, data, target, optimizer=None):
        '''The terms of the loss function along with their coefficients.

        Args:
            model: An `nn.Module` or a function to process `data`.
            data: The desired input to `model` (e.g., a batch of images).
            target: The desired output of `model` (e.g., ground truth labels).
            optimizer: The optimizer.

        Returns:
            (terms: A `dict` of `Namespace`s that contains
             (coef, func, args, kwargs), term = coef * func(*args, **kwargs).
             If `cls.ignore_zero_loss_coefficients is True` and coef == 0,
             the call of func() will be ingored and never executed.
             metrics: Some collected metrics.)
        '''
        raise NotImplementedError

    @classmethod
    def process(cls, model, data, target, optimizer=None):
        '''Process a single batch and update the weights if desired.

        Args:
            model: An `nn.Module` or a function to process `data`.
            data: The desired input to `model` (e.g., a batch of images).
            target: The desired output of `model` (e.g., ground truth labels).
            optimizer: If not provided, the model will not be updated.

        Returns:
            A `dict` of metrics.
        '''
        values = []
        terms, metrics = cls.loss(model, data, target, optimizer)
        for key, term in terms.items():
            if not (cls.ignore_zero_loss_coefficients and term.coef == 0):
                value = term.func(*term.args, **term.kwargs)
                metrics['loss/' + key] = float(value) * len(data)
                values.append(float(term.coef) * value)
        total_loss = sum(values)
        metrics['loss'] = float(total_loss) * len(data)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return metrics

    @classmethod
    def full_epoch(cls, model, data_loader, device, optimizer=None):
        '''Perform a single epoch.

        Args:
            model: An `nn.Module` or a function to process `data`.
            data_loader: A 'torch.utils.data.DataLoader'.
            device: On which device to perfrom the epoch.

        Returns:
            A `dict` of collected metrics.
        '''
        # Change model.training to True and False accordingly
        model.eval() if optimizer is None else model.train()
        model.to(device)
        total_count = 0
        accumulated_metrics = {}
        for data, target in verbosify(data_loader):
            # process the batch [data (images) and target (labels)]
            metrics = cls.process(
                model, data.to(device), target.to(device), optimizer)
            # accumlate the metrics
            total_count += len(data)
            for metric, value in metrics.items():
                if metric not in accumulated_metrics:
                    accumulated_metrics[metric] = 0
                accumulated_metrics[metric] += value
        # compute the averaged metrics
        for metric in accumulated_metrics:
            accumulated_metrics[metric] /= total_count
        return accumulated_metrics

    @classmethod
    def train(cls, model, device, num_epochs, optimizer, train_loader,
              valid_loader, scheduler=None, patience=10, load=None,
              save=None, log_dir=None, restart=False):
        '''Train a model for a certain number of epochs.

        Args:
            model: A function to process the batches form the loaders.
            device: In which device to do the training.
            num_epochs: Number of epochs to train.
            optimizer: An `torch.optim.Optimizer` (e.g. SGD).
            train_loader: The `DataLoader` for the training dataset.
            valid_loader: The `DataLoader` for the validation dataset.
            scheduler: The learning rate scheduler.
            patience: The number of bad epochs to wait before early stopping.
            load: Load the model from this `*.pt` checkpoint file.
            save: The `*.pt` checkpoint file to save all the trained model.
            log_dir: The directory to save tensorboard summaries.
            restart: Whether to remove `log_dir` and `load` before training.

        Returns:
            The best state of the model during training (maximum valid loss).
        '''
        # restart if desired by removing old files
        if restart:
            if log_dir is not None and os.path.exists(log_dir):
                rmtree(log_dir)
            if load is not None and os.path.exists(load):
                os.remove(load)

        # try to resume from a checkpoint file if `load` was provided
        if load is not None:
            try:
                best_state = torch.load(load)
                model.load_state_dict(best_state['model'])
                optimizer.load_state_dict(best_state['optimizer'])
                scheduler.load_state_dict(best_state['scheduler'])
            except FileNotFoundError:
                msg = ('Couldn\'t find checkpoint file! {} '
                       '(training with random initialization)')
                print(msg.format(load))
                load = None

        # otherwise, start from the current initialization
        if load is None:
            best_state = {
                'epoch': -1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': float('inf'),
            }

        # create the directory to the checkpoint file if it doens't exists
        if save is not None:
            save_dir = os.path.dirname(save)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        model.to(device)
        num_bad_epochs = 0
        msg = ('Epoch #{}: [train: {:.2e} > {:.2f}%]'
               '[valid: {:.2e} > {:.2f}%] @ {:.2e}')
        for epoch in range(best_state['epoch'] + 1, num_epochs):
            # train and validate
            train_metrics = cls.full_epoch(
                model, train_loader, device, optimizer)
            valid_metrics = cls.full_epoch(
                model, valid_loader, device)  # will not do backward pass

            # get the current learing rate
            learning_rate = optimizer.param_groups[0]['lr']
            # Note: an nn.Module can have multiple param_groups
            #       each of which can be assigned a different learning rate
            #       but by default we have a single param_group.

            # reduce the learning rate according to the `scheduler` policy
            if scheduler is not None:
                scheduler.step(valid_metrics['loss'])

            # print the progress
            print(msg.format(
                epoch, train_metrics['loss'], 100 * train_metrics['accuracy'],
                valid_metrics['loss'], 100 * valid_metrics['accuracy'],
                learning_rate,
            ))

            # save tensorboard summaries
            if None not in (log_dir, importlib.util.find_spec('tensorflow')):
                tf = importlib.import_module('tensorflow')

                # create the summary writer only the first time
                if not hasattr(log_dir, 'add_summary'):
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    log_dir = tf.summary.FileWriter(log_dir)
                summaries = {
                    'learning_rate': learning_rate,
                }
                summaries.update({'train/' + name: value
                                  for name, value in train_metrics.items()})
                summaries.update({'valid/' + name: value
                                  for name, value in valid_metrics.items()})
                values = [tf.Summary.Value(tag=k, simple_value=v)
                          for k, v in summaries.items()]
                log_dir.add_summary(tf.Summary(value=values), epoch)
                log_dir.flush()

            # save the model to disk if it has improved
            is_nan = valid_metrics['loss'] != valid_metrics['loss']
            if is_nan or best_state['loss'] < valid_metrics['loss']:
                num_bad_epochs += 1
            else:
                num_bad_epochs = 0
                best_state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': valid_metrics['loss'],
                }
                if save is not None:
                    torch.save(best_state, save)

            # do early stopping
            if is_nan or num_bad_epochs >= patience:
                msg = 'Validation loss didn\'t improve for {} iterations!'
                if is_nan:
                    msg = 'Validation loss reached NaN!'
                print(msg.format(patience))
                print('[Early stopping]')
                break

        # close the summary writer if created
        if log_dir is not None:
            if hasattr(log_dir, 'close'):
                log_dir.close()

        return best_state

    @classmethod
    def model_from_config(cls, config, load_checkpoint=False):
        model = config.model.network(**vars(config.model.config))
        model = model.to(config.device)
        model.config = config
        if load_checkpoint and os.path.exists(config.checkpoint):
            state = torch.load(config.checkpoint)['model']
            model.load_state_dict(state)
        return model

    @classmethod
    def data_from_config(cls, config, train=False, model=None):
        if model is None:
            model = cls.model_from_config(config, load_checkpoint=False)
        if train:
            data = config.data
            dataset = data.train.dataset(**vars(data.train.config))
            train_set, valid_set = cls.random_split(
                dataset, data.valid.split,
                train_transform=data.train.transform(model),
                train_target_transform=data.train.target_transform(model),
                valid_transform=data.valid.transform(model),
                valid_target_transform=data.valid.target_transform(model))

            cuda = torch.device(config.device).type == 'cuda'
            train_loader = cls.data_loader(train_set, data.train.batch_size,
                                           train=True, cuda=cuda,
                                           workers=data.train.num_loaders)
            valid_loader = cls.data_loader(valid_set, data.valid.batch_size,
                                           train=False, cuda=cuda,
                                           workers=data.valid.num_loaders)
            return train_loader, valid_loader
        else:
            data = config.data.test
            test_set = TransformedDataset.of(
                data.dataset(**vars(data.config)),
                transform=data.transform(model),
                target_transform=data.target_transform(model))

            cuda = torch.device(config.device).type == 'cuda'
            loader = cls.data_loader(test_set, data.batch_size,
                                     train=False, cuda=cuda,
                                     workers=data.num_loaders)
            return loader

    @classmethod
    def train_from_config(cls, config):
        old_ignore = cls.ignore_zero_loss_coefficients
        try:
            model = cls.model_from_config(config, load_checkpoint=False)

            opt = config.optimization
            cls.ignore_zero_loss_coefficients = opt.loss_terms.ignore_zeros
            optimizer = opt.optimizer(model.parameters(), **vars(opt.config))

            sched = config.lr_scheduling
            scheduler = sched.scheduler(optimizer, **vars(sched.config))

            train, valid = cls.data_from_config(config, True, model)
            load_checkpoint = config.checkpoint if config.finetune else None
            best_state = cls.train(model, config.device,
                                   config.epochs, optimizer,
                                   train, valid,
                                   scheduler=scheduler,
                                   patience=config.patience,
                                   save=config.checkpoint,
                                   load=load_checkpoint,
                                   log_dir=config.log_dir,
                                   restart=config.restart)
            return best_state
        finally:
            cls.ignore_zero_loss_coefficients = old_ignore

    @classmethod
    def test(cls, config):
        model = cls.model_from_config(config, load_checkpoint=True)
        performance = getattr(model, config.model.metric_function)
        loader = cls.data_from_config(config, model=model, train=False)
        return performance(loader, config.device)

    @classmethod
    def default_config(cls):
        config = Namespace(
            epochs=30,
            patience=10,
            finetune=True,
            restart=False,
            device='cuda',
            optimization=Namespace(
                optimizer=torch.optim.Adam,
                config=Namespace(
                    lr=1e-4,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                ),
                loss_terms=Namespace(
                    empirical=1,
                    ignore_zeros=True,
                ),
            ),
            lr_scheduling=Namespace(
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                config=Namespace(
                    mode='min',
                    factor=0.9,
                    patience=3,
                    verbose=True,
                ),
            ),
        )
        return config

    @classmethod
    def config_as_dict(cls, config):
        if not isinstance(config, Namespace):
            return config
        return {k: cls.config_as_dict(v) for k, v in vars(config).items()}

    @classmethod
    def random_split(cls, dataset, split_frac=0.1,
                     train_transform=None, train_target_transform=None,
                     valid_transform=None, valid_target_transform=None):
        '''Split a given dataset randomly to training and validation subsets.

        Args:
            dataset: The dataset to split.
            split_frac: The validation ratio.
            train_transform: The transform for the train subset.
            train_target_transform: The transform for the target train subset.
            valid_transform: The transform for the valid subset.
            valid_target_transform: The transform for the target valid subset.

        Returns:
            (train_set, valid_set)
        '''
        dataset_length = len(dataset)
        train_length = int(dataset_length * (1 - split_frac))
        valid_length = dataset_length - train_length
        train_set, valid_set = torch.utils.data.random_split(
            dataset, [train_length, valid_length])
        train_set = TransformedDataset.of(
            train_set, transform=train_transform,
            target_transform=train_target_transform)
        valid_set = TransformedDataset.of(
            valid_set, transform=valid_transform,
            target_transform=valid_target_transform)
        return train_set, valid_set

    @classmethod
    def data_loader(cls, dataset, batch_size,
                    train=False, cuda=False, workers=4):
        '''Wraps torch.utils.data.DataLoader with the given params.'''
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           pin_memory=cuda,
                                           num_workers=workers if cuda else 0,
                                           shuffle=train,
                                           drop_last=train)


class ClassifierTrainer(Trainer):
    '''Trainer for classifiers.

    Example:
    class CustomClassifierTrainer(ClassifierTrainer):
        @classmethod
        def loss(cls, model, data, target, optimizer):
            terms, metrics = super().loss(model, data, target, optimizer)
            for i in range(model.num_classes):
                metrics[f'class_frequency/{i}'] = (target == i).sum().item()
            return terms, metrics

        @classmethod
        def default_config(cls):
            config = super().default_config()
            cls.config_model_dataset(config, 'lenet', 'mnist')
            config.log_dir = 'exps/mnist/lenet/poc'
            config.checkpoint = 'models/mnist/lenet_poc.pt'
            config.data.train.batch_size = 5000
            return config


    config = CustomClassifierTrainer.default_config()
    config.device = 'cuda:1'
    config.restart = True
    best_state = CustomClassifierTrainer.train_from_config(config)
    print(f'Test accuracy = {100 * CustomClassifierTrainer.test(config):.2f}%')
    '''
    network = {
        'lenet': LeNet,
        'alexnet': AlexNet,
        'vgg16': VGG16,
    }
    dataset = {
        'mnist': Namespace(
            input_size=28,
            num_classes=10,
            type=torchvision.datasets.MNIST,
            input_mean=(0.5,),
            input_std=(1,),
        ),
        'cifar10': Namespace(
            input_size=32,
            num_classes=10,
            type=torchvision.datasets.CIFAR10,
            input_mean=(0.4915, 0.4823, 0.4468),
            input_std=(0.2470, 0.2435, 0.2616),
        ),
        'cifar100': Namespace(
            input_size=32,
            num_classes=100,
            type=torchvision.datasets.CIFAR100,
            input_mean=(0.5072, 0.4867, 0.4412),
            input_std=(0.2673, 0.2564, 0.2762),
        ),
    }

    @classmethod
    def loss(cls, model, data, target, optimizer):
        phase = torch.no_grad if optimizer is None else torch.enable_grad
        with phase():
            output = model(data)
        terms = {
            'empirical': Namespace(coef=1, func=cls.softmax_cross_entropy,
                                   args=(output, target), kwargs={})
        }
        metrics = {
            'accuracy': cls.count_correct(output.data, target).item()
        }
        return terms, metrics

    @staticmethod
    def count_correct(output, target):
        predictions = output.max(1, keepdim=True)[1]
        return predictions.eq(target.view_as(predictions)).sum()

    @staticmethod
    def softmax_cross_entropy(output, target):
        F = torch.nn.functional
        # more efficient than `F.cross_entropy(F.softmax(output), target)`
        return F.nll_loss(F.log_softmax(output, dim=1), target)

    @classmethod
    def config_model_dataset(cls, config,
                             model='lenet', dataset='mnist', name='default'):
        '''Add model and dataset specific configurations to `config`.

        Args:
            config: The configurations to edit.
            model: The name of the model.
            dataset: The name of the dataset.
            name: The name of the configuration.
        '''
        dataset_config = cls.dataset[dataset]
        config.data = Namespace(
            train=Namespace(batch_size=1000),
            valid=Namespace(batch_size=5000, split=0.1),
            test=Namespace(batch_size=5000)
        )
        for key, set_config in vars(config.data).items():
            if key in {'train', 'test'}:
                set_config.dataset = dataset_config.type
                set_config.config = Namespace(
                    root=f'data/{dataset}',
                    download=True,
                    train=key == 'train',
                )
            set_config.num_loaders = 4
            set_config.transform = lambda net: net.default_transform()
            set_config.target_transform = lambda net: None

        kwargs = {k: v for k, v in vars(dataset_config).items() if k != 'type'}
        config.model = Namespace(network=cls.network[model],
                                 config=Namespace(**kwargs),
                                 metric_function='accuracy')

        config.log_dir = f'exps/{dataset}/{model}/{name}'
        config.checkpoint = f'models/{dataset}/{model}_{name}.pt'
