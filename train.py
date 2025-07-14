import os
import sys
import math
import random
import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from transformers import Adafactor
from pytorch_lamb import Lamb
import wandb


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#Deterministic randomness as a controlled variable during experimentation
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

warnings.filterwarnings("ignore", category=UserWarning)

#DATASET FOR TRAINING AND EVALUATION
DATASET = "Augmented_Dataset.parquet"

#K-Fold Cross Examination Switch
HPTUNING = False
OTUNING = False
if HPTUNING:
    K = 5
else:
    K = 0
if not HPTUNING:
    if OTUNING:
        print("To activate OTUNING, HPTUNING must also be active to allow for experimentation with and the comparison of different optimisers")
        sys.exit()

#DATA LOADER PARAMETERS
TRAIN_VAL_RATIO = 0.8 #static, ratio of training and validation datasets
PADDING_VALUE = 0.0 #static, padding with zeroes
BATCH_SIZE = 32 #samples per training loop, larger batch sizes allow for efficient training at the cost of generalisation and memory usage. Smaller batch sizes lead to better generalisation at the cost of slower training, higher variance and less stable convergence

#TRANSFORMER PARAMETERS
INPUT_DIM = 126 #static, feature vector size
D_MODEL = 64 #dimensionality of token embeddings and internal representations. A larger model dimension allows for more complex patterns to be captured at the cost of training efficiency and a risk of overfitting, being too small can lead to low accuracy and a weaker ability to capture complex patterns.
NUM_HEADS = 4 #attend to different parts of the model in parallel, too few will limit the models ability to capture diverse relationships whilst too many can dilute learning per head leading to no benefit/detriment to performance at the cost of increased overhead
NUM_LAYERS = 3 #too few limits the models ability to learn patters from features in higher layers, too many can cause overfitting, vanishing gradients and lead to inefficient training
FF_DIM_MULT = D_MODEL * 4 #static, hidden size of feedforward layer
MEM_LEN = 40 #cached timesteps for recurrence, too few limits context increases memory consumption, slows computation and may cause overfitting
DROPOUT = 0.3 #used to prevent overfitting by zeroing activations during training, too low may won't help prevent overfitting, too high could discard too many useful signals, hindering learning

#TRAINING AND VALIDATION PARAMETERS
SAVE_PATH = "best_model.pt"
NUM_EPOCHS = 50 #times the model trains over the whole dataset, too few leads to underfitting, too many could lead to overfitting
PATIENCE_LIMIT = 2 #number of consecutive runs without improvement of the validation loss before early stopping is applied to prevent overfitting
LEARNING_RATE = 1e-4 #step size for updating parameters, too high causes instability and divergence, too low leads to slow convergence and risks getting stuck at a poor minima 
WEIGHT_DECAY = 1e-4 #adds L2 regularisation to reduce overfitting by penalising large weights, too low doesn't offer benefit, too high causes underfitting
MIN_DELTA = 1e-3 #threshold for runs that are considered to improve performance, small values can lead to overfitting through extended training, large values can ignore the benefit from small yet useful improvements
NUM_CLASSES = 2 #static, correct and incorrect gestures (binary)


# --- Configuration for initialising weights and biases to log metrics ---
def init_wandb():
    return wandb.init(
        entity="x-ix-transformer",
        project="Mediapipe Matrix Gesture Recognition K-Fold",
        name = f"BS {BATCH_SIZE}, DROP {DROPOUT}, DIM {D_MODEL}, H {NUM_HEADS}, L {NUM_LAYERS}",
        config={
            "architecture": "Transformer (encoder only)",
            "dataset": "Custom 78/78 mediapipe matrix set for binary classification",

            #DATA LOADING PARAMETERS
            "batch_size" : BATCH_SIZE,

            #TRANSFORMER PARAMETERS
            "input_dimensions" : INPUT_DIM,
            "model_dimensionality" : D_MODEL,
            "attention_heads" : NUM_HEADS,
            "model_layers" : NUM_LAYERS,
            "feedforward_layers" : FF_DIM_MULT,
            "memory_length" : MEM_LEN,
            "dropout" : DROPOUT,

            #TRAINING AND VALIDATION PARAMETERS
            "epochs" : NUM_EPOCHS,
            "patience_limit" : PATIENCE_LIMIT,
            "learning_rate" : LEARNING_RATE,
            "weight_decay" : WEIGHT_DECAY
        },
    )



# --- Inspect parquet file to inform dataset and dataloader configuration ---
def inspect_parquet(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    print("DataFrame columns:", df.columns)
    print("\n--- First row ---")
    print(df.iloc[0])
    print("\n--- Type of 'Combined' column first element ---")
    print(type(df['Combined'].iloc[0]))
    print("\n--- Content of first element ---")
    print(df['Combined'].iloc[0])
    print("\n--- Shape if it can be converted ---")
    try:
        arr = np.array(df['Combined'].iloc[0])
        print("Shape:", arr.shape)
    except Exception as e:
        print("Failed to get shape:", e)



# --- Custom Dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences: List[torch.Tensor], labels: List[torch.Tensor]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]



# --- Load Dataset from Parquet ---
def load_dataset_from_parquet(parquet_path: str) -> TimeSeriesDataset:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    df = pd.read_parquet(parquet_path)

    # Shuffle DataFrame deterministically
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Separate sequences and labels
    all_seqs = [
        torch.stack([torch.tensor(row, dtype=torch.float32) for row in sequence])
        for sequence in df['Combined']
    ]
    labels = [int(lbl) for lbl in df['label']]
    label_tensors = [torch.tensor(lbl, dtype=torch.long) for lbl in labels]

    #return TimeSeriesDataset(normed_seqs, label_tensors)
    return TimeSeriesDataset(all_seqs, label_tensors)



# --- Split Dataset into Train and Val sets, conduct K-Fold splitting if required ---
def split_dataset(dataset: Dataset, k=0, iteration=0, train_ratio: float = TRAIN_VAL_RATIO):
    # Group indices by label (to balance True/False instances in each set)
    label_0_indices = [i for i in range(len(dataset)) if dataset[i][1].item() == 0]
    label_1_indices = [i for i in range(len(dataset)) if dataset[i][1].item() == 1]

    #if not K-Fold, shuffle and split normally, if K-Fold do not shuffle, split based on iteration (current fold)
    def stratified_split(indices, train_ratio, k=0, iteration=0):
        if k == 0:
            random.shuffle(indices)
            split = int(len(indices) * train_ratio)
            return indices[:split], indices[split:]
        else:
            if len(indices) % k != 0:
                print("equal divisions of dataset required")
                sys.exit()
            fold_size = len(indices) // k
            val_start = iteration * fold_size
            val_end = val_start + fold_size
            val = indices[val_start:val_end]
            train = indices[:val_start] + indices[val_end:]
            return train, val

    # Perform stratified split on both classes
    train_0, val_0 = stratified_split(label_0_indices, train_ratio, k, iteration)
    train_1, val_1 = stratified_split(label_1_indices, train_ratio, k, iteration)

    #combine the two labelled sets for train and validation respectively
    train_indices = train_0 + train_1
    val_indices = val_0 + val_1

    # Shuffle combined indices to avoid label ordering
    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
    


# --- Normalise Train then apply the mean and std to Val  ---
def normalise(train_dataset, val_dataset, tuning: bool = False):
    # Stack all sequences from training set and calculate mean/std
    train_data = torch.cat([x[0] for x in train_dataset], dim=0)
    mean = train_data.mean(dim=0)
    std = train_data.std(dim=0) + 1e-8  # avoid division by zero

    def apply_norm(dataset):
        norm_seqs = [( (x - mean) / std, y ) for x, y in dataset]
        return norm_seqs

    #using function defined above, normalise the train set based on it's mean and std, then normalise the validation set based on train's mean and std to prevent data leakage
    norm_train = apply_norm(train_dataset)
    norm_val = apply_norm(val_dataset)

    #if not hyperparameter tuning, save normalisation configuration for inference
    if not tuning:
        torch.save({'mean': mean, 'std': std}, 'normalisation_stats.pt')

    return norm_train, norm_val



# --- Collate Function ---
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    # Prepares batches by padding sequences to equal length and creating masks
    sequences, labels = zip(*batch)
    lengths = [seq.size(0) for seq in sequences]
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=PADDING_VALUE)
    max_len = padded_seqs.size(1)
    pad_mask = torch.arange(max_len).expand(len(sequences), max_len) >= torch.tensor(lengths).unsqueeze(1)
    return padded_seqs, pad_mask, torch.tensor(labels), torch.tensor(lengths)



# --- Create DataLoaders ---
def create_dataloaders(train_dataset, val_dataset, batch_size: int = BATCH_SIZE):
    #Data loading configurations for the train and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader



# --- Manual F1 Score Calculation ---
def f1_score(true_labels: list[int], pred_labels: list[int]) -> float:
    """Compute binary F1 score from lists of 0/1 labels."""
    #true positives, false positives, false negatives
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)

    #precision factors in false positives whilst recall factors in false negatives
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    #calculates the harmonic mean of precision and recall
    return 2 * (precision * recall) / (precision + recall)



# --- Update memory for each layer ---
def update_memory(hidden_states: List[torch.Tensor], memory: List[torch.Tensor], mem_len: int) -> List[torch.Tensor]:
    # If no existing memory, return the most recent mem_len hidden states for each layer
    if memory is None:
        return [h.detach()[-mem_len:] for h in hidden_states]
    else:
        updated_memory = []
        # For each layer, concatenate old memory and new hidden states, then keep the most recent mem_len entries
        for h, m in zip(hidden_states, memory):
            cat = torch.cat([m, h.detach()], dim=0)
            updated_memory.append(cat[-mem_len:])
        return updated_memory



# --- Computes attention using relative positions ---
class RelMultiheadAttention(nn.Module):
    def __init__(self, d_model = D_MODEL, nhead = NUM_HEADS, mem_len = MEM_LEN, dropout= DROPOUT):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.mem_len = mem_len
        
        # projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # relative positional embeddings
        self.rel_emb = nn.Parameter(torch.randn(2 * mem_len - 1, self.d_head))
        
        # biases u and v
        self.u = nn.Parameter(torch.randn(self.nhead, self.d_head))
        self.v = nn.Parameter(torch.randn(self.nhead, self.d_head))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, memory_length=0):
        # query: [T, B, D], key/value: [(M+T), B, D]
        T, B, D = query.size()
        S = key.size(0)
        M = memory_length

        # project and reshape: Q->[T,B,h,d], K,V->[S,B,h,d]
        Q = self.q_proj(query).view(T, B, self.nhead, self.d_head)
        K = self.k_proj(key).view(S, B, self.nhead, self.d_head)
        V = self.v_proj(value).view(S, B, self.nhead, self.d_head)

        # relative position indices
        # distance = (s - M) - t -> ranges [-(M-1), T-1]
        dist = torch.arange(S, device=query.device).unsqueeze(1) - torch.arange(T, device=query.device).unsqueeze(0) - M
        idxs = (dist + (self.mem_len - 1)).clamp(0, 2 * self.mem_len - 2)

        # fetch relative embeddings: [T, S, d_head]
        R = self.rel_emb[idxs.view(-1)].view(T, S, self.d_head)

        # content-based scores: [B,h,T,S]
        AC = torch.einsum('t b h d, s b h d -> b h t s', Q, K)
        # content-to-position scores: [B,h,T,S]
        AP = torch.einsum('t b h d, t s d -> b h t s', Q, R)

        # u term: bias for keys -> [B,h,1,S]
        UB = torch.einsum('h d, s b h d -> b h s', self.u, K).unsqueeze(2)
        # v term: bias for positions -> [1,h,T,S]
        VB = torch.einsum('h d, t s d -> h t s', self.v, R).unsqueeze(0)

        # combine and scale
        scores = (AC + AP + UB + VB) / math.sqrt(self.d_head)

        # apply masks
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # compute attention output: [T,B,D]
        out = torch.einsum('b h t s, s b h d -> t b h d', attn, V)
        out = out.contiguous().view(T, B, D)
        return self.out_proj(out)


# --- Transformer-XL style encoder layer enabling temporal dependency learning across segments ---
class TransformerXLEncoderLayer(nn.Module):
    def __init__(self, d_model = D_MODEL, nhead = NUM_HEADS, dim_feedforward = FF_DIM_MULT, mem_len = MEM_LEN, dropout = DROPOUT):
        super().__init__()
        # Multi-head self-attention with relative positional encoding
        self.self_attn = RelMultiheadAttention(d_model, nhead, mem_len, dropout)
        # Feedforward network layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Layer normalizations and additional dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, memory=None, src_mask=None, src_key_padding_mask=None):
        T, B, D = src.size()
        # Concatenate past memory if provided
        if memory is not None:
            src_combined = torch.cat([memory, src], dim=0)
            mem_len = memory.size(0)
        else:
            src_combined = src
            mem_len = 0

        # Compute self-attention output
        attn_output = self.self_attn(
            query=src,
            key=src_combined,
            value=src_combined,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            memory_length=mem_len
        )

        # Add & normalize attention output
        src2 = src + self.dropout1(attn_output)
        src2 = self.norm1(src2)

        # Feedforward network and residual connection
        ff = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        out = src2 + self.dropout2(ff)
        out = self.norm2(out)
        return out

    

# --- stack of encoder layers with memory propagation across segments ---
class TransformerXLStackedEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FF_DIM_MULT, num_layers=NUM_LAYERS, dropout=DROPOUT, mem_len=MEM_LEN):
        super().__init__()
        self.mem_len = mem_len
        self.layers = nn.ModuleList([TransformerXLEncoderLayer(d_model, nhead, dim_feedforward, mem_len, dropout)for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None, memories=None):
        T, B, _ = x.shape
        if pad_mask is None:
            pad_mask = x.new_zeros(B, T, dtype=torch.bool)
        if memories is None:
            memories = [None] * len(self.layers)
        new_memories = []
        for mem, layer in zip(memories, self.layers):

            # build padding mask that covers both memory and current input
            if mem is not None:
                M = mem.size(0)
                mem_mask = torch.zeros(B, M, dtype=torch.bool, device=x.device)
                combined_mask = torch.cat([mem_mask, pad_mask], dim=1)
            else:
                combined_mask = pad_mask

            # apply the layer
            x = layer(src=x, memory=mem, src_key_padding_mask=combined_mask)

            # update memory: if no prior mem, just take the last mem_len of x
            if mem is None:
                new_mem = x.detach()[-self.mem_len:]
            else:
                new_mem = update_memory([x], [mem], mem_len=self.mem_len)[0]
            new_memories.append(new_mem)

        return self.norm(x), new_memories  



# --- Project each frame’s 126‑dim feature vector into model dimension ---
class FrameEmbedding(nn.Module):
    def __init__(self, feature_dim: int = INPUT_DIM, d_model: int = D_MODEL, dropout: float = DROPOUT):
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, feature_dim]
        Returns:
            emb: [T, B, d_model]
        """
        # project features
        emb = self.proj(x)
        # apply dropout
        return self.dropout(emb)



# --- Classification Head ---
class ClassificationHead(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        # Single logit for binary classification
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, D]  (encoder output)
        Returns:
            logits: [B]   (one logit per sequence)
        """
        # Take the last time step
        last_hidden = x[-1]            # [B, D]
        logits = self.fc(last_hidden)  # [B, 1]
        return logits.squeeze(-1)      # [B]



# # --- Transformer Classifier ---
class TransformerXLClassifier(nn.Module):
    def __init__(self, feature_dim: int = INPUT_DIM, d_model: int = D_MODEL, nhead: int = NUM_HEADS, dim_feedforward: int = FF_DIM_MULT, num_layers: int = NUM_LAYERS, mem_len: int = MEM_LEN, dropout: float = DROPOUT):
        super().__init__()
        self.mem_len = mem_len
        self.embedding = FrameEmbedding(feature_dim, d_model, dropout)
        self.encoder   = TransformerXLStackedEncoder(d_model, nhead, dim_feedforward, num_layers, dropout)
        self.head      = ClassificationHead(d_model)

    def init_memory(self, batch_size: int, device: torch.device):
        return [
            torch.zeros(self.mem_len, batch_size, self.embedding.proj.out_features, device=device)
            for _ in range(len(self.encoder.layers))
        ]

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None, memories: list[torch.Tensor] = None):
        """
        Args:
            x: [B, T, feature_dim]
            pad_mask: [B, T] boolean mask (True = padding)
            memories: list of [M, B, D] or None
        Returns:
            logits: [B]
            new_memories: list of [M, B, D]
        """
        B, T, _ = x.shape

        # Embed to [T, B, D]
        emb = self.embedding(x.transpose(0, 1))  # [T, B, D]

        # Forward through encoder with padding mask
        enc_out, new_memories = self.encoder(emb, pad_mask=pad_mask, memories=memories)

        # Classification head
        logits = self.head(enc_out)  # [B]
        return logits, new_memories



# --- Training and Validation Loops ---
def train_model(model, train_loader, val_loader, device, run, save_path=SAVE_PATH, num_epochs=NUM_EPOCHS, patience_limit=PATIENCE_LIMIT, opt = None):
    # Use BCEWithLogits for a single‐logit binary head
    criterion = nn.BCEWithLogitsLoss()

    #in the case of optimiser testing
    if opt:
        optimizer = opt
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        # ——— Training phase ———
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1} [Training]",
                         leave=False)
        for inputs, pad_mask, labels, lengths in train_bar:
            # inputs: [B, T, F], pad_mask: [B, T], labels: [B]
            B = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            # reset memory per sequence batch
            memories = model.init_memory(batch_size=B, device=device)


            optimizer.zero_grad()
            # pass pad_mask into the model
            logits, _ = model(inputs, pad_mask=pad_mask.to(device), memories=memories)


            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)

        # ——— Validation phase ———
        model.eval()
        val_loss = 0.0
        all_preds  = []
        all_labels = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False)
        with torch.no_grad():
            for inputs, pad_mask, labels, lengths in val_bar:
                B = inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                # reset memory per validation batch
                memories = model.init_memory(batch_size=B, device=device)

                logits, _ = model(inputs, pad_mask=pad_mask.to(device), memories=memories)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # threshold at 0.5
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.long().cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_f1  = f1_score(all_labels, all_preds)

        # log metrics
        run.log({
            "epoch":    epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss":   avg_val_loss,
            "val_acc":    val_acc,
            "val_f1":     val_f1
        })
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        # early stopping
        if best_val_loss - avg_val_loss > MIN_DELTA:
            best_val_loss = avg_val_loss
            patience = 0
            if not HPTUNING:
                torch.save(model.state_dict(), save_path)
                print("Saved best model.")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"No improvement for {patience_limit} epochs, stopping.")
                break

    print("Training complete.")
    if HPTUNING:
        return {"val_loss": avg_val_loss, "val_acc": val_acc, "val_f1": val_f1}
    


# --- main ---
def main():
    tuning = HPTUNING
    optim = OTUNING
    k = K

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset_from_parquet(DATASET)

    if not tuning:
        # Single run: initialise wandb, prepare data, train model, log results
        run = init_wandb()

        model = TransformerXLClassifier(feature_dim=INPUT_DIM, d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FF_DIM_MULT, num_layers=NUM_LAYERS, mem_len=MEM_LEN, dropout=DROPOUT).to(device)

        train_ds, val_ds = split_dataset(dataset, k=0, iteration=0, train_ratio=TRAIN_VAL_RATIO)
        norm_train, norm_val = normalise(train_ds, val_ds)

        train_loader, val_loader = create_dataloaders(norm_train, norm_val, batch_size=BATCH_SIZE)

        train_model(model, train_loader, val_loader, device, run, save_path=SAVE_PATH, num_epochs=NUM_EPOCHS, patience_limit=PATIENCE_LIMIT)
        run.finish()

    else:
        if not optim:
            # K-fold tuning: repeat training across k folds, collect performance metrics
            val_losses = []
            val_accs = []
            val_f1s    = []

            for i in range(k):
                run = init_wandb()

                model = TransformerXLClassifier(feature_dim=INPUT_DIM, d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FF_DIM_MULT, num_layers=NUM_LAYERS, mem_len=MEM_LEN, dropout=DROPOUT).to(device)

                train_ds, val_ds = split_dataset(dataset, k=k, iteration=i, train_ratio=TRAIN_VAL_RATIO)
                norm_train, norm_val = normalise(train_ds, val_ds, tuning)

                train_loader, val_loader = create_dataloaders(norm_train, norm_val, batch_size=BATCH_SIZE)

                metrics = train_model(model, train_loader, val_loader, device, run, save_path=f"{SAVE_PATH}.fold{i}", num_epochs=NUM_EPOCHS, patience_limit=PATIENCE_LIMIT)
                val_losses.append(metrics["val_loss"])
                val_accs.append(metrics["val_acc"])
                val_f1s.append(metrics["val_f1"])
                run.finish()
            
            # Report mean validation performance across folds
            print("K_Fold cross evaluation results:")
            print(f"Mean validation loss over {k} folds: {np.mean(val_losses):.4f}")
            print(f"Mean validation accuracy over {k} folds: {np.mean(val_accs):.4f}")
            print(f"Mean validation F1 score over {k} folds: {np.mean(val_f1s):.4f}")
        else:
            #cycles through different optimisers on three model configurations to determine an optimal choice
            optimisers = {
                "Adam": lambda params: torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
                "LAMB": lambda params: Lamb(params, lr=3e-3, weight_decay=1e-2, betas=(0.9, 0.999)),
                "Adafactor": lambda params: Adafactor(params, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
            }

            configs = [
                {"name": "small", "d_model": 32, "num_heads": 1, "num_layers": 1, "dropout": 0.1},
                {"name": "medium", "d_model": 64, "num_heads": 2, "num_layers": 2, "dropout": 0.3},
                {"name": "large", "d_model": 128, "num_heads": 4, "num_layers": 3, "dropout": 0.3}
            ]

            results = {}

            for config in configs:
                print(f"\nTesting config: {config['name']}")

                for opt_name, opt_func in optimisers.items():
                    val_losses, val_accs, val_f1s = [], [], []
                    print(f"  Running K-Fold with {opt_name}...")

                    for i in range(k):
                        run = wandb.init(mode="disabled")

                        ff_mult = config["d_model"] * 4

                        model = TransformerXLClassifier(
                            feature_dim=INPUT_DIM,
                            d_model=config["d_model"],
                            nhead=config["num_heads"],
                            dim_feedforward=ff_mult,
                            num_layers=config["num_layers"],
                            mem_len=MEM_LEN,
                            dropout=config["dropout"]
                        ).to(device)

                        train_ds, val_ds = split_dataset(dataset, k=k, iteration=i, train_ratio=TRAIN_VAL_RATIO)
                        norm_train, norm_val = normalise(train_ds, val_ds, tuning=True)
                        train_loader, val_loader = create_dataloaders(norm_train, norm_val, batch_size=BATCH_SIZE)

                        opt = opt_func(model.parameters())

                        metrics = train_model(
                            model, train_loader, val_loader, device, run,
                            save_path=f"{SAVE_PATH}.{config['name']}.{opt_name}.fold{i}",
                            num_epochs=NUM_EPOCHS, patience_limit=PATIENCE_LIMIT, opt=opt
                        )

                        val_losses.append(metrics["val_loss"])
                        val_accs.append(metrics["val_acc"])
                        val_f1s.append(metrics["val_f1"])
                        run.finish()

                    results[(config["name"], opt_name)] = {
                        "loss": np.mean(val_losses),
                        "acc": np.mean(val_accs),
                        "f1": np.mean(val_f1s)
                    }

        
            print("\nOptimizer Comparison (Mean Across Folds per Config):")
            print(f"{'Config':<8} | {'Optimizer':<10} | {'Loss':<10} | {'Accuracy':<10} | {'F1 Score':<10}")
            print("-" * 60)

            for opt_name in sorted(optimisers.keys()):
                for config in configs:
                    key = (config["name"], opt_name)
                    metrics = results.get(key)
                    if metrics:
                        print(f"{config['name']:<8} | {opt_name:<10} | {metrics['loss']:<10.4f} | {metrics['acc']:<10.4f} | {metrics['f1']:<10.4f}")


if __name__ == "__main__":
    main()
