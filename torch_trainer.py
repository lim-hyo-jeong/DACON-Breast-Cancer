import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch_dataset import BCDataset, BCMILDataset
from torch_model import BCModel, BCMILModel
from utils import get_scheduler
from metrics import macro_f1
from transforms import train_transforms, test_transforms
from copy import deepcopy
import settings
import gc
from tqdm import tqdm
import time


class BCTrainer:

    def __init__(self, config, logger):
        self.config = config
        self.model_params = config['model_params']
        self.training_params = config['training_params']
        self.inference_params = config['inference_params']
        self.transform_params = config['transform_params']
        self.logger = logger

    def train(self, dataloader, model, criterion, optimizer, scheduler, epoch, device):
        start_time = time.time()
        train_loss = []
        total_labels, total_preds = [], []

        lr = scheduler.get_last_lr()[
            0] if scheduler is not None and self.training_params['scheduler'] != 'ReduceLROnPlateau' else optimizer.param_groups[0]['lr']

        model.train()
        scaler = GradScaler()

        for images, tabulars, labels in tqdm(dataloader):
            images = images.to(device)
            tabulars = tabulars.to(device)
            labels = labels.to(device)

            with autocast():
                optimizer.zero_grad()
                preds = model(images, tabulars)

                loss = criterion(preds, labels)
                train_loss.append(loss.detach().item())

                total_labels += [(labels.detach().cpu())]
                total_preds += [(preds.detach().cpu())]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and self.training_params['scheduler'] != 'ReduceLROnPlateau':
                scheduler.step()

        train_loss = np.mean(train_loss)

        total_labels = torch.cat(total_labels, dim=0).numpy()
        total_preds = torch.cat(total_preds, dim=0).numpy()
        train_score = macro_f1(total_labels, np.where(total_preds > 0.5, 1, 0))

        elapsed = time.time() - start_time
        self.logger.info(
            f'[Epoch {epoch}] train_loss : {train_loss:.6f} | train_macro_f1 : {train_score:.6f} | lr : {lr:.8f} | time : {elapsed:.0f}s')

        return train_loss, train_score

    def validate(self, dataloader, model, criterion, scheduler, epoch, device):
        start_time = time.time()
        valid_loss = []
        total_labels, total_preds = [], []

        model.eval()

        for images, tabulars, labels in tqdm(dataloader):
            images = images.to(device)
            tabulars = tabulars.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(images, tabulars)

                loss = criterion(preds, labels)
                valid_loss.append(loss.detach().item())

                total_labels += [(labels.detach().cpu())]
                total_preds += [(preds.detach().cpu())]

        valid_loss = np.mean(valid_loss)

        total_labels = torch.cat(total_labels, dim=0).numpy()
        total_preds = torch.cat(total_preds, dim=0).numpy()
        valid_score = macro_f1(total_labels, np.where(total_preds > 0.5, 1, 0))

        if self.training_params['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(
                valid_loss if self.training_params['monitor'] == 'loss' else valid_score)

        elapsed = time.time() - start_time
        self.logger.info(
            f'[Epoch {epoch}] valid_loss : {valid_loss:.6f} | valid_macro_f1 : {valid_score:.6f} | time : {elapsed:.0f}s')

        return valid_loss, valid_score

    def train_and_validate(self, df):
        best_loss_recorder = []
        best_score_recorder = []

        model_save_dir = settings.MODEL / self.model_params['model_save_dir']
        model_save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info('Run training !')
        self.logger.info(f'config : {self.config}')

        for fold in range(self.config['n_folds']):
            self.logger.info(f'Fold {fold} training ...')

            df_train = df[df['fold'] != fold]
            df_valid = df[df['fold'] == fold]
            tabular_features = self.config['tabular_features']
            target = self.training_params['target']
            device = self.training_params['device']

            train_dataset = BCDataset(
                ids=df_train['ID'].values,
                tabulars=df_train[tabular_features].values,
                labels=df_train[target].values,
                transforms=train_transforms(self.transform_params),
            )

            valid_dataset = BCDataset(
                ids=df_valid['ID'].values,
                tabulars=df_valid[tabular_features].values,
                labels=df_valid[target].values,
                transforms=test_transforms(self.transform_params),
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_params['batch_size'],
                shuffle=True,
                drop_last=False,
                num_workers=self.training_params['num_workers'],
                pin_memory=False,
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.training_params['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=self.training_params['num_workers'],
                pin_memory=False,
            )

            model = BCModel(
                backbone_name=self.model_params['backbone_name'],
                pretrained=True,
                backbone_out_dims=self.model_params['backbone_out_dims'],
                n_tabulars=len(tabular_features),
                tabular_out_dims=self.model_params['tabular_out_dims'],
                n_classes=self.config['n_classes']).to(device)

            criterion = nn.BCEWithLogitsLoss().to(device)
            optimizer = optim.AdamW(
                model.parameters(), lr=self.training_params['lr'])
            scheduler = get_scheduler(optimizer, self.training_params)

            best_loss = np.inf
            best_score = 0.
            best_epoch = -1
            best_model_weights = deepcopy(model.state_dict())
            patience = 0
            start_time = time.time()

            for epoch in range(self.training_params['epochs']):
                if patience == self.training_params['early_stopping_rounds']:
                    self.logger.info(
                        f'No improvement since epoch {epoch-patience-1}: early stopping')
                    break

                train_loss, train_score = self.train(
                    train_loader, model, criterion, optimizer, scheduler, epoch, device)
                valid_loss, valid_score = self.validate(
                    valid_loader, model, criterion, scheduler, epoch, device)
                self.logger.info('-' * 100)

                if self.training_params['monitor'] == 'loss':
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        best_score = valid_score
                        best_epoch = epoch
                        best_model_weights = deepcopy(model.state_dict())
                        patience = 0
                    else:
                        patience += 1
                elif self.traning_params['monitor'] == 'score':
                    if valid_score > best_score:
                        best_score = valid_score
                        best_loss = valid_loss
                        best_epoch = epoch
                        best_model_weights = deepcopy(model.state_dict())
                        patience = 0
                    else:
                        patience += 1

            model.load_state_dict(best_model_weights)
            model_save_path = model_save_dir / f'model_f{fold}_best.pth'
            torch.save(model.state_dict(), model_save_path)

            best_loss_recorder.append(best_loss)
            best_score_recorder.append(best_score)

            elapsed = time.time() - start_time
            self.logger.info(
                f'[Fold {fold}] best_epoch : {best_epoch} | valid_loss : {best_loss} | valid_macro_f1 :{best_score} | time : {elapsed:.0f}s')
            self.logger.info(
                f'[Fold {fold}] best model saved : {model_save_path}')

            del model
            torch.cuda.empty_cache()
            gc.collect()

        self.logger.info('#' * 100)
        self.logger.info('#' * 100)
        self.logger.info(
            f'nAverage best valid_loss : {np.mean(best_loss_recorder):.6f}')
        self.logger.info(
            f'Average best valid_macro_f1 : {np.mean(best_score_recorder):.6f}')

    def inference(self, df_test):
        final_preds = []

        tabular_features = self.config['tabular_features']
        device = self.inference_params['device']

        test_dataset = BCDataset(
            ids=df_test['ID'].values,
            tabulars=df_test[tabular_features].values,
            labels=None,
            transforms=test_transforms(self.transform_params),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.inference_params['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.inference_params['num_workers'],
            pin_memory=False,
        )

        for fold in range(self.config['n_folds']):
            start_time = time.time()
            total_preds = []

            model = BCModel(
                backbone_name=self.model_params['backbone_name'],
                pretrained=False,
                backbone_out_dims=self.model_params['backbone_out_dims'],
                n_tabulars=len(tabular_features),
                tabular_out_dims=self.model_params['tabular_out_dims'],
                n_classes=self.config['n_classes']).to(device)

            model_save_path = settings.MODEL / \
                self.model_params['model_save_dir'] / f'model_f{fold}_best.pth'
            model.load_state_dict(torch.load(model_save_path))
            model.eval()

            for images, tabulars in tqdm(test_loader):
                images = images.to(device)
                tabulars = tabulars.to(device)

                with torch.no_grad():
                    preds = model(images, tabulars)
                    total_preds += [(preds.detach().cpu())]

            total_preds = torch.cat(total_preds, dim=0).numpy()
            final_preds.append(total_preds)

            elapsed = time.time() - start_time
            self.logger.info(
                f'[model_f{fold}_best] inference time : {elapsed:.0f}s')

            del model
            torch.cuda.empty_cache()
            gc.collect()

        final_preds = np.mean(np.stack(final_preds), axis=0)
        preds_save_path = settings.MODEL / \
            self.model_params['model_save_dir'] / f'preds.npy'
        np.save(preds_save_path, final_preds)
        self.logger.info(
            f'Prediction result saved : {preds_save_path}')

    def save_oof(self, df):
        final_oof = np.zeros((df.shape[0], 1))

        for fold in range(self.config['n_folds']):
            start_time = time.time()
            df_valid = df[df['fold'] == fold]
            valid_idx = df_valid.index
            tabular_features = self.config['tabular_features']
            target = self.training_params['target']
            device = self.training_params['device']

            valid_dataset = BCDataset(
                ids=df_valid['ID'].values,
                tabulars=df_valid[tabular_features].values,
                labels=df_valid[target].values,
                transforms=test_transforms(self.transform_params),
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.inference_params['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=self.inference_params['num_workers'],
                pin_memory=False,
            )

            model = BCModel(
                backbone_name=self.model_params['backbone_name'],
                pretrained=False,
                backbone_out_dims=self.model_params['backbone_out_dims'],
                n_tabulars=len(tabular_features),
                tabular_out_dims=self.model_params['tabular_out_dims'],
                n_classes=self.config['n_classes']).to(device)

            model_save_path = settings.MODEL / \
                self.model_params['model_save_dir'] / f'model_f{fold}_best.pth'
            model.load_state_dict(torch.load(model_save_path))
            model.eval()

            criterion = nn.BCEWithLogitsLoss().to(device)
            valid_loss = []
            total_labels, total_preds = [], []

            for images, tabulars, labels in tqdm(valid_loader):
                images = images.to(device)
                tabulars = tabulars.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(images, tabulars)

                    loss = criterion(preds, labels)
                    valid_loss.append(loss.detach().item())

                    total_labels += [(labels.detach().cpu())]
                    total_preds += [(preds.detach().cpu())]

            valid_loss = np.mean(valid_loss)

            total_labels = torch.cat(total_labels, dim=0).numpy()
            total_preds = torch.cat(total_preds, dim=0).numpy()
            final_oof[valid_idx] = total_preds
            total_preds = np.where(total_preds > 0.5, 1, 0)
            valid_score = macro_f1(total_labels, total_preds)

            elapsed = time.time() - start_time
            self.logger.info(
                f'[model_f{fold}_best] valid_loss : {valid_loss:.6f} | valid_macro_f1 : {valid_score:.6f} | time : {elapsed:.0f}s')

            del model
            torch.cuda.empty_cache()
            gc.collect()

        oof_save_path = settings.MODEL / \
            self.model_params['model_save_dir'] / f'oof.npy'
        np.save(oof_save_path, final_oof)
        self.logger.info(
            f'Validation result saved : {oof_save_path}')


class BCMILTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.model_params = config['model_params']
        self.training_params = config['training_params']
        self.inference_params = config['inference_params']
        self.transform_params = config['transform_params']
        self.logger = logger

    def train(self, dataloader, model, criterion, optimizer, scheduler, epoch, device):
        start_time = time.time()
        train_loss = []
        total_labels, total_preds = [], []

        lr = scheduler.get_last_lr()[
            0] if scheduler is not None and self.training_params['scheduler'] != 'ReduceLROnPlateau' else optimizer.param_groups[0]['lr']

        model.train()
        scaler = GradScaler()

        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                optimizer.zero_grad()
                preds = model(images)

                loss = criterion(preds, labels)
                train_loss.append(loss.detach().item())

                total_labels += [(labels.detach().cpu())]
                total_preds += [(preds.detach().cpu())]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and self.training_params['scheduler'] != 'ReduceLROnPlateau':
                scheduler.step()

        train_loss = np.mean(train_loss)

        total_labels = torch.cat(total_labels, dim=0).numpy()
        total_preds = torch.cat(total_preds, dim=0).numpy()
        train_score = macro_f1(total_labels, np.where(total_preds > 0.5, 1, 0))

        elapsed = time.time() - start_time
        self.logger.info(
            f'[Epoch {epoch}] train_loss : {train_loss:.6f} | train_macro_f1 : {train_score:.6f} | lr : {lr:.8f} | time : {elapsed:.0f}s')

        return train_loss, train_score

    def validate(self, dataloader, model, criterion, scheduler, epoch, device):
        start_time = time.time()
        valid_loss = []
        total_labels, total_preds = [], []

        model.eval()

        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(images)

                loss = criterion(preds, labels)
                valid_loss.append(loss.detach().item())

                total_labels += [(labels.detach().cpu())]
                total_preds += [(preds.detach().cpu())]

        valid_loss = np.mean(valid_loss)

        total_labels = torch.cat(total_labels, dim=0).numpy()
        total_preds = torch.cat(total_preds, dim=0).numpy()
        valid_score = macro_f1(total_labels, np.where(total_preds > 0.5, 1, 0))

        if self.training_params['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(
                valid_loss if self.training_params['monitor'] == 'loss' else valid_score)

        elapsed = time.time() - start_time
        self.logger.info(
            f'[Epoch {epoch}] valid_loss : {valid_loss:.6f} | valid_macro_f1 : {valid_score:.6f} | time : {elapsed:.0f}s')

        return valid_loss, valid_score

    def train_and_validate(self, df):
        best_loss_recorder = []
        best_score_recorder = []

        model_save_dir = settings.MODEL / self.model_params['model_save_dir']
        model_save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info('Run training !')
        self.logger.info(f'config : {self.config}')

        for fold in range(self.config['n_folds']):
            self.logger.info(f'Fold {fold} training ...')

            df_train = df[df['fold'] != fold]
            df_valid = df[df['fold'] == fold]
            target = self.training_params['target']
            device = self.training_params['device']

            train_dataset = BCMILDataset(
                ids=df_train['ID'].values,
                n_instances=self.model_params['n_instances'],
                labels=df_train[target].values,
                transforms=train_transforms(self.transform_params),
            )

            valid_dataset = BCMILDataset(
                ids=df_valid['ID'].values,
                n_instances=self.model_params['n_instances'],
                labels=df_valid[target].values,
                transforms=test_transforms(self.transform_params),
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_params['batch_size'],
                shuffle=True,
                drop_last=False,
                num_workers=self.training_params['num_workers'],
                pin_memory=False,
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.training_params['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=self.training_params['num_workers'],
                pin_memory=False,
            )

            model = BCMILModel(
                backbone_name=self.model_params['backbone_name'],
                pretrained=True,
                backbone_out_dims=self.model_params['backbone_out_dims'],
                n_instances=self.model_params['n_instances'],
                n_classes=self.config['n_classes']).to(device)

            criterion = nn.BCEWithLogitsLoss().to(device)
            optimizer = optim.AdamW(
                model.parameters(), lr=self.training_params['lr'])
            scheduler = get_scheduler(optimizer, self.training_params)

            best_loss = np.inf
            best_score = 0.
            best_epoch = -1
            best_model_weights = deepcopy(model.state_dict())
            patience = 0
            start_time = time.time()

            for epoch in range(self.training_params['epochs']):
                if patience == self.training_params['early_stopping_rounds']:
                    self.logger.info(
                        f'No improvement since epoch {epoch-patience-1}: early stopping')
                    break

                train_loss, train_score = self.train(
                    train_loader, model, criterion, optimizer, scheduler, epoch, device)
                valid_loss, valid_score = self.validate(
                    valid_loader, model, criterion, scheduler, epoch, device)
                self.logger.info('-' * 100)

                if self.training_params['monitor'] == 'loss':
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        best_score = valid_score
                        best_epoch = epoch
                        best_model_weights = deepcopy(model.state_dict())
                        patience = 0
                    else:
                        patience += 1
                elif self.traning_params['monitor'] == 'score':
                    if valid_score > best_score:
                        best_score = valid_score
                        best_loss = valid_loss
                        best_epoch = epoch
                        best_model_weights = deepcopy(model.state_dict())
                        patience = 0
                    else:
                        patience += 1

            model.load_state_dict(best_model_weights)
            model_save_path = model_save_dir / f'model_f{fold}_best.pth'
            torch.save(model.state_dict(), model_save_path)

            best_loss_recorder.append(best_loss)
            best_score_recorder.append(best_score)

            elapsed = time.time() - start_time
            self.logger.info(
                f'[Fold {fold}] best_epoch : {best_epoch} | valid_loss : {best_loss} | valid_macro_f1 :{best_score} | time : {elapsed:.0f}s')
            self.logger.info(
                f'[Fold {fold}] best model saved : {model_save_path}')

            del model
            torch.cuda.empty_cache()
            gc.collect()

        self.logger.info('#' * 100)
        self.logger.info('#' * 100)
        self.logger.info(
            f'nAverage best valid_loss : {np.mean(best_loss_recorder):.6f}')
        self.logger.info(
            f'Average best valid_macro_f1 : {np.mean(best_score_recorder):.6f}')

    def inference(self, df_test):
        final_preds = []

        device = self.inference_params['device']

        test_dataset = BCMILDataset(
            ids=df_test['ID'].values,
            n_instances=self.model_params['n_instances'],
            labels=None,
            transforms=test_transforms(self.transform_params),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.inference_params['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.inference_params['num_workers'],
            pin_memory=False,
        )

        for fold in range(self.config['n_folds']):
            start_time = time.time()
            total_preds = []

            model = BCMILModel(
                backbone_name=self.model_params['backbone_name'],
                pretrained=False,
                backbone_out_dims=self.model_params['backbone_out_dims'],
                n_instances=self.model_params['n_instances'],
                n_classes=self.config['n_classes']).to(device)

            model_save_path = settings.MODEL / \
                self.model_params['model_save_dir'] / f'model_f{fold}_best.pth'
            model.load_state_dict(torch.load(model_save_path))
            model.eval()

            for images in tqdm(test_loader):
                images = images.to(device)

                with torch.no_grad():
                    preds = model(images)
                    total_preds += [(preds.detach().cpu())]

            total_preds = torch.cat(total_preds, dim=0).numpy()
            final_preds.append(total_preds)

            elapsed = time.time() - start_time
            self.logger.info(
                f'[model_f{fold}_best] inference time : {elapsed:.0f}s')

            del model
            torch.cuda.empty_cache()
            gc.collect()

        final_preds = np.mean(np.stack(final_preds), axis=0)
        preds_save_path = settings.MODEL / \
            self.model_params['model_save_dir'] / f'preds.npy'
        np.save(preds_save_path, final_preds)
        self.logger.info(
            f'Prediction result saved : {preds_save_path}')

    def save_oof(self, df):
        final_oof = np.zeros((df.shape[0], 1))

        for fold in range(self.config['n_folds']):
            start_time = time.time()
            df_valid = df[df['fold'] == fold]
            valid_idx = df_valid.index
            target = self.training_params['target']
            device = self.training_params['device']

            valid_dataset = BCMILDataset(
                ids=df_valid['ID'].values,
                n_instances=self.model_params['n_instances'],
                labels=df_valid[target].values,
                transforms=test_transforms(self.transform_params),
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.inference_params['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=self.inference_params['num_workers'],
                pin_memory=False,
            )

            model = BCMILModel(
                backbone_name=self.model_params['backbone_name'],
                pretrained=False,
                backbone_out_dims=self.model_params['backbone_out_dims'],
                n_instances=self.model_params['n_instances'],
                n_classes=self.config['n_classes']).to(device)

            model_save_path = settings.MODEL / \
                self.model_params['model_save_dir'] / f'model_f{fold}_best.pth'
            model.load_state_dict(torch.load(model_save_path))
            model.eval()

            criterion = nn.BCEWithLogitsLoss().to(device)
            valid_loss = []
            total_labels, total_preds = [], []

            for images, labels in tqdm(valid_loader):
                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(images)

                    loss = criterion(preds, labels)
                    valid_loss.append(loss.detach().item())

                    total_labels += [(labels.detach().cpu())]
                    total_preds += [(preds.detach().cpu())]

            valid_loss = np.mean(valid_loss)

            total_labels = torch.cat(total_labels, dim=0).numpy()
            total_preds = torch.cat(total_preds, dim=0).numpy()
            final_oof[valid_idx] = total_preds
            total_preds = np.where(total_preds > 0.5, 1, 0)
            valid_score = macro_f1(total_labels, total_preds)

            elapsed = time.time() - start_time
            self.logger.info(
                f'[model_f{fold}_best] valid_loss : {valid_loss:.6f} | valid_macro_f1 : {valid_score:.6f} | time : {elapsed:.0f}s')

            del model
            torch.cuda.empty_cache()
            gc.collect()

        oof_save_path = settings.MODEL / \
            self.model_params['model_save_dir'] / f'oof.npy'
        np.save(oof_save_path, final_oof)
        self.logger.info(
            f'Validation result saved : {oof_save_path}')
