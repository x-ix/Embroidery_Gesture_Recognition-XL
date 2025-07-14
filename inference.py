import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import mediapipe as mp
import numpy as np
import shutil
import math
import time
import cv2

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

#trained model
inference_model = "best_model.pt"

#mediapipe config
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#mediapipe Constants
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MS_IN_SECOND = 1000
HANDS = 1

# Comment out if integrating streamed input
video_path_1 = "Test_Case/Top-View.mp4"
video_path_2 = "Test_Case/Bottom-View.mp4"

#TRANSFORMER PARAMETERS
INPUT_DIM = 126 #static, feature vector size
D_MODEL = 64 #dimensionality of token embeddings and internal representations. A larger model dimension allows for more complex patterns to be captured at the cost of training efficiency and a risk of overfitting, being too small can lead to low accuracy and a weaker ability to capture complex patterns.
NUM_HEADS = 4 #attend to different parts of the model in parallel, too few will limit the models ability to capture diverse relationships whilst too many can dilute learning per head leading to no benefit/detriment to performance at the cost of increased overhead
NUM_LAYERS = 3 #too few limits the models ability to learn patters from features in higher layers, too many can cause overfitting, vanishing gradients and lead to inefficient training
FF_DIM_MULT = D_MODEL * 4 #static, hidden size of feedforward layer
MEM_LEN = 40 #cached timesteps for recurrence, too few limits context increases memory consumption, slows computation and may cause overfitting
DROPOUT = 0.3 #used to prevent overfitting by zeroing activations during training, too low may won't help prevent overfitting, too high could discard too many useful signals, hindering learning

#mean and std, import to apply to input dim before inference
stats = torch.load('normalisation_stats.pt')
mean, std = stats['mean'], stats['std']



# --- cv2 capture initialisation and fps retrieval ---
def initialise_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps



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



# --- Get preditions from model ---
class Predictor:
    def __init__(self, model: TransformerXLClassifier, device: torch.device, mem_len: int):
        self.model = model.to(device)
        self.device = device
        self.memories = self.model.init_memory(batch_size=1, device=device)  # batch size = 1 for real-time
        self.mem_len = mem_len

    def update(self, input_tensor: torch.Tensor) -> tuple[int, float]:
        """
        Args:
            input_tensor: [1, feature_dim] → a single frame (already normalized)
        Returns:
            predicted class (0 or 1), probability of class 1
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.unsqueeze(1).to(self.device)  # shape: [1, 1, feature_dim] → T=1, B=1
            logits, self.memories = self.model(input_tensor, memories=self.memories)
            prob = torch.sigmoid(logits)[0].item()
            pred_class = int(prob > 0.5)
        return pred_class, prob



# --- main ---
def main():
    # --- Loading model for inference ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerXLClassifier().to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    
    predictor = Predictor(model, device, mem_len=MEM_LEN)



    # --- Mediapipe model --
    model = model_path
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=HANDS,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1
    )



    #Setting up two capture devices to simulate two input streams
    cap1, fps1 = initialise_video_capture(video_path_1)
    cap2, fps2 = initialise_video_capture(video_path_2)

    Avg_time = []

    with HandLandmarker.create_from_options(options) as landmarker1, \
     HandLandmarker.create_from_options(options) as landmarker2:
        
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break

            frame_number1 = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
            frame_number2 = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame1)
            mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2)

            timestamp_ms1 = ((frame_number1 / fps1) * MS_IN_SECOND) - (MS_IN_SECOND / fps1)
            timestamp_ms2 = ((frame_number2 / fps2) * MS_IN_SECOND) - (MS_IN_SECOND / fps2)


            start_time = time.time()
            detection_result1 = landmarker1.detect_for_video(mp_image1, int(timestamp_ms1))
            detection_result2 = landmarker2.detect_for_video(mp_image2, int(timestamp_ms2))

            if detection_result1.hand_landmarks and detection_result2.hand_landmarks:
                hand1 = detection_result1.hand_landmarks[0]  # assuming one hand
                hand2 = detection_result2.hand_landmarks[0]  # assuming one hand

                #Handlandmarks come in the form (1, 3) for 21 landmarks resulting in a matrix of (21, 3) which is then flattened to (1,63)
                matrix_entry1 = np.array([[lm.x, lm.y, lm.z] for lm in hand1], dtype=np.float32).flatten()
                matrix_entry2 = np.array([[lm.x, lm.y, lm.z] for lm in hand2], dtype=np.float32).flatten()

                # Convert to tensors without copying
                tensor1 = torch.from_numpy(matrix_entry1).unsqueeze(0)  # shape (1, 63)
                tensor2 = torch.from_numpy(matrix_entry2).unsqueeze(0)  # shape (1, 63)

                # Concatenate (Top-View on left concatenated with Bottom-View on right) to from a matrix of shape (1, 63 + 63) = (1, 126)
                combined = torch.cat([tensor1, tensor2], dim=1)  # shape (1, 126)
                normalized = ((combined - mean) / std).to(dtype=torch.float32, device=device)

                
                pred_class, prob1 = predictor.update(normalized)
                end_time = time.time()

                inference_time_ms = (end_time - start_time) * 1000  

                Avg_time.append(inference_time_ms)
                
                print(f"Prediction: {pred_class}, P(class=1): {prob1:.4f}, Time: {inference_time_ms:.2f} ms")

    print('-' * shutil.get_terminal_size().columns)
    print(f"Average time for mediapipe labelling and transformer classification: {np.mean(Avg_time[1:])} ms")

    cap1.release()
    cap2.release()
 

if __name__ == "__main__":
    main()








