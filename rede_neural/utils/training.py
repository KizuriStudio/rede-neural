import numpy as np
import time
import sys
import os


class LRScheduler:
    """Schedulers de learning rate."""

    @staticmethod
    def constant(lr, epoch):
        return lr

    @staticmethod
    def step(lr, epoch, step_size=10, gamma=0.1):
        return lr * (gamma ** (epoch // step_size))

    @staticmethod
    def cosine_annealing(lr, epoch, max_epochs, min_lr=1e-6):
        return min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs))

    @staticmethod
    def warmup_cosine(lr, epoch, warmup_epochs=5, max_epochs=100, min_lr=1e-6):
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * progress))

    @staticmethod
    def linear_warmup(lr, epoch, warmup_epochs=5, factor=1.0):
        if epoch < warmup_epochs:
            return lr * factor * (epoch + 1) / warmup_epochs
        return lr


class EarlyStopping:
    """Para o treino quando a loss para de diminuir."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class Trainer:
    """Loop de treino completo com logging bonito no terminal."""

    def __init__(self, model, loss_fn, optimizer, scheduler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        n_batches = 0

        for x_batch, y_batch in dataloader:
            output = self.model.forward(x_batch)
            loss = self.loss_fn.forward(output.reshape(-1, output.shape[-1]), y_batch.flatten())

            grad = self.loss_fn.backward()
            self.model.backward(grad)

            grad_norm = self._clip_gradients(max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        n_batches = 0

        for x_batch, y_batch in dataloader:
            output = self.model.forward(x_batch)
            loss = self.loss_fn.forward(output.reshape(-1, output.shape[-1]), y_batch.flatten())
            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(self, train_loader, val_loader=None, epochs=100, lr=0.001, early_stopping=None, gen_fn=None, gen_every=5):
        start_lr = lr
        best_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        self._print_header(epochs, len(train_loader.dataset))

        for epoch in range(epochs):
            epoch_start = time.time()

            current_lr = lr
            if self.scheduler:
                if self.scheduler == "cosine":
                    current_lr = LRScheduler.cosine_annealing(start_lr, epoch, epochs)
                elif self.scheduler == "warmup_cosine":
                    current_lr = LRScheduler.warmup_cosine(start_lr, epoch, warmup_epochs=max(1, epochs // 10), max_epochs=epochs)
                elif self.scheduler == "step":
                    current_lr = LRScheduler.step(start_lr, epoch)
                self.optimizer.lr = current_lr

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader) if val_loader else train_loss

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start
            self._print_epoch(epoch + 1, epochs, train_loss, val_loss, current_lr, elapsed)

            if gen_fn and (epoch + 1) % gen_every == 0:
                print()
                gen_fn(self.model)
                print()

            if val_loss < best_loss:
                best_loss = val_loss

            if early_stopping:
                early_stopping(val_loss)
                if early_stopping.should_stop:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

        self._print_footer(best_loss)

    def _clip_gradients(self, max_norm=1.0):
        total_norm = 0
        grads = self.optimizer.grads
        for k in grads:
            if grads[k] is not None:
                total_norm += np.sum(grads[k] ** 2)
        total_norm = np.sqrt(total_norm)

        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for k in grads:
                if grads[k] is not None:
                    grads[k] *= clip_coef

        return total_norm

    def _print_header(self, epochs, dataset_size):
        print("=" * 65)
        print("  REDE NEURAL — TREINO DE LANGUAGE MODEL")
        print("=" * 65)
        print(f"  Epochs: {epochs} | Dados: {dataset_size} amostras")
        print(f"  Otimizador: {self.optimizer.__class__.__name__}")
        print(f"  Loss: {self.loss_fn.__class__.__name__}")
        print("-" * 65)
        print(f"  {'Ep':>4}  {'Train Loss':>11}  {'Val Loss':>9}  {'LR':>10}  {'Tempo':>6}")
        print("-" * 65)

    def _print_epoch(self, epoch, total, train_loss, val_loss, lr, elapsed):
        bar_len = 20
        filled = int(bar_len * epoch / total)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"\r  {epoch:>4}/{total}  {train_loss:>9.4f}  {val_loss:>9.4f}  "
            f"{lr:>10.2e}  {elapsed:>5.1f}s  {bar}",
            end="", flush=True
        )

    def _print_footer(self, best_loss):
        print("\n" + "=" * 65)
        print(f"  Treino completo! Melhor val loss: {best_loss:.4f}")
        print(f"  Perplexity: {np.exp(min(best_loss, 20)):.2f}")
        print("=" * 65)
