#*********** Necessary Libraries ******************
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import time
import copy



#*********** Set random seed ********************
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                        # covers CPU + MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # each worker gets a different, but deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(tensor_dataset, batch_size, shuffle=True, seed=123):
    gen = torch.Generator()
    gen.manual_seed(seed)             
    return torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,         
        worker_init_fn=seed_worker,
        num_workers=0
    )



#*********** Helping Functions ********************
class Autoencoder(nn.Module):
    """
    Autoencoder: It compresses inputs to a low-dimensional latent representation (encoder)
                 and reconstructs them back to the input space (decoder).

    Parameters
    ----------
    input_dim : int
        Number of input features.
    latent_dim : int
        Size of the latent code.
    hidden1    : int
        Size of the encoder first layer.
    hideen2.   : int
        Size of the encoder second layer.
    activation : str
        Nonlinearity for hidden layers.
    
    Returns
    --------
    x_hat      : torch.Tensor
        Reconstruction of the inputs.
    """

    def __init__(self, input_dim=42, latent_dim=10, hidden1=64, hidden2=32, activation='Mish'):
        super(Autoencoder, self).__init__()
        act_fn = getattr(nn, activation)
        # Encoder (inputdim - 64 - 32 - 10)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            act_fn(),
            nn.Linear(hidden1, hidden2),
            act_fn(),
            nn.Linear(hidden2, latent_dim)
            )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            act_fn(),
            nn.Linear(hidden2, hidden1),
            act_fn(),
            nn.Linear(hidden1, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    




def train_step(model, train_loader, optimizer, device, val_loader=None,
               epochs=500, patience=10, min_delta=1e-4, verbose_every=5
               ):
    """
    Train an autoencoder with optional early stopping on validation loss.

    Parameters
    ----------
    model : nn.Module
        Autoencoder model.
    train_loader : DataLoader
        Dataloader for training data.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : torch.device
        Device for computation.
    val_loader : DataLoader, optional
        Dataloader for validation data.
    epochs : int, default=500
        Maximum number of epochs.
    patience : int, default=10
        Number of epochs to wait for validation improvement.
    min_delta : float, default=1e-4
        Minimum improvement required to reset patience.
    verbose_every : int, default=5
        Print progress every N epochs.

    Returns
    -------
    model : nn.Module
        Trained model with best validation weights loaded if val_loader is provided.
    train_history : list[float]
        Training loss history per epoch.
    val_history : list[float]
        Validation loss history per epoch.
    """

    model = model.to(device)

    train_history = []
    val_history = []

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    no_improve = 0
    converged_epoch = epochs

    epoch_times = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Training step
        model.train()
        train_sample_errs = []

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)

            optimizer.zero_grad()

            x_hat = model(x_batch)

            # batch loss used for optimization
            loss = F.mse_loss(x_hat, x_batch, reduction="mean")
            loss.backward()
            optimizer.step()

            # per-sample MSE for consistent monitoring
            sample_err = torch.mean((x_batch - x_hat) ** 2, dim=1)
            train_sample_errs.append(sample_err.detach().cpu().numpy())

        train_sample_errs = np.concatenate(train_sample_errs)
        train_loss = float(np.mean(train_sample_errs))
        train_history.append(train_loss)

        # Validation step and early stopping
        if val_loader is not None:
            model.eval()
            val_sample_errs = []

            with torch.no_grad():
                for (x_batch,) in val_loader:
                    x_batch = x_batch.to(device)
                    x_hat = model(x_batch)

                    sample_err = torch.mean((x_batch - x_hat) ** 2, dim=1)
                    val_sample_errs.append(sample_err.cpu().numpy())

            val_sample_errs = np.concatenate(val_sample_errs)
            val_loss = float(np.mean(val_sample_errs))
            val_history.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1

            if epoch % verbose_every == 0:
                print(f"Epoch {epoch:03d}: Train {train_loss:.6f}, Val {val_loss:.6f}")

            if no_improve >= patience:
                converged_epoch = epoch
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch: {best_epoch}, best val: {best_val_loss:.6f})"
                )
                break

        else:
            if epoch % verbose_every == 0:
                print(f"Epoch {epoch:03d}: Train {train_loss:.6f}")

        epoch_times.append(time.time() - start_time)

    # Load best model state if validation was used
    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)

    total_time = sum(epoch_times)
    avg_epoch_time = total_time / len(epoch_times)

    print("=" * 60)
    print(f"Training complete in {total_time:.2f}s over {len(train_history)} epochs")
    print(f"Average epoch time: {avg_epoch_time:.2f}s")
    if val_loader is not None:
        print(f"Converged at epoch: {converged_epoch}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)

    return model, train_history, val_history

    

def exponential_smoothing(x, alpha=0.2):
    """
    Apply exponential smoothing to each column of a 2D array.
    x: shape (n_samples, n_features)
    alpha: smoothing factor (0 < alpha < 1)
    """
    x = np.asarray(x)
    smoothed = np.zeros_like(x)
    smoothed[0] = x[0]  # initialize with first row

    for t in range(1, x.shape[0]):
        smoothed[t] = alpha * x[t] + (1 - alpha) * smoothed[t - 1]

    return smoothed



def threshold_from_far(pre_residuals, target_far=0.0035):
    """
    Compute threshold for a residual/statistic series to achieve a target FAR.
    
    Parameters
    ----------
    pre_residuals : array-like
        1D array of residuals/statistics from calibration (fault-free) data
    target_far : float
        Desired false alarm rate (e.g., 0.01 for 1%)
        
    Returns
    -------
    threshold : float
        Value above which the residual/statistic is considered a fault
    """
    # Percentile corresponding to the FAR
    percentile = 100 * (1 - target_far)
    threshold = np.percentile(pre_residuals, percentile)
    return threshold




def compute_fdr_far_mdr(true_labels, pred_labels):
    """
    Compute the fault detection metrics

    Parameters:
    -----------
        true_labels: list or array of the ground truth labels (0 for normal, 1 for faulty)
        pred_labels: list or array of predicted fault labels (0 for normal, 1 for faulty)
    
    Returns:
    --------
        fdr:         Fault detection rate
        far:         False alarm rate
        mdr:         Missed detection rate (1-fdr)
        acc:         Accuracy

    """
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

    fdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Fault Detection Rate
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate
    # mdr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # Missed Detection Rate
    acc = (tp + tn) / (tp + fn + tn + fp) if (tp + fn + tn + fp) > 0 else 0.0  # Accuracy

    return fdr, far, acc



def get_detection_delay(prediction_labels, fault_start_idx, ts=1):
    """
    Compute the time delay in fault detection

    Parameters:
    -----------
        prediction_labels:  list or array of predicted fault labels (0 for normal, 1 for faulty)
        fault_start_idx:    Index where fault starts
        ts:                 Sampling time (per minute/seconds of samples)

    Returns:
    ---------
        detection_delay:    Delay in samples (or -1 if not detected)
        time_to_detection:  Delay in time units (or -1 if not detected)
    """
    detection_idx = None
    for idx in range(fault_start_idx, len(prediction_labels)):
        if prediction_labels[idx] == 1:
            detection_idx = idx
            break

    if detection_idx is not None:
        detection_delay = detection_idx - fault_start_idx
        time_to_detection = detection_delay * ts
    else:
        detection_delay = -1
        time_to_detection = -1

    return detection_delay, time_to_detection

