import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, rate=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(rate)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(), 
            nn.Dropout(rate),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(rate)
        )
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # MultiheadAttention expects (Seq_Len, Batch, Dim)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(attn_output + x)
        ffn_output = self.ffn(out1)
        out2 = self.layer_norm2(ffn_output + out1)
        return out2

class TextureContrastClassifier(nn.Module):
    def __init__(self, input_shape=(128, 256), num_heads=4, ff_dim=256, rate=0.2):
        super(TextureContrastClassifier, self).__init__()
        input_dim = input_shape[1] 
        seq_len = input_shape[0]   
        
        self.rich_attention = AttentionBlock(input_dim, num_heads, ff_dim, rate)
        self.poor_attention = AttentionBlock(input_dim, num_heads, ff_dim, rate)
        
        self.spectral_reducer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(rate)
        )

        self.visual_cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten()
        )

        fusion_dim = (seq_len * 64 * 2) + 2048
        
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Defense against False Positives
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Sigmoid removed for BCEWithLogitsLoss
        )

    def forward(self, rich, poor, ela, noise):
        r = self.rich_attention(rich.permute(1, 0, 2))
        p = self.poor_attention(poor.permute(1, 0, 2))
        
        r_feat = self.spectral_reducer(r.permute(1, 0, 2)).reshape(rich.size(0), -1)
        p_feat = self.spectral_reducer(p.permute(1, 0, 2)).reshape(poor.size(0), -1)
        
        if ela.dim() == 3:
            ela = ela.unsqueeze(-1)
        
        v_in = torch.cat([ela.permute(0, 3, 1, 2), noise.unsqueeze(1)], dim=1)
        v_feat = self.visual_cnn(v_in)
        
        combined = torch.cat([r_feat, p_feat, v_feat], dim=1)
        return self.fc(combined)