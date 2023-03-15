import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import stft, istft


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              groups=in_channels, stride=stride, padding=padding, bias=bias)
    def forward(self, x):
        x = self.conv(x)
        return x

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding='same', bias=True):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=stride, padding=padding, bias=bias)
    def forward(self, x):
        x = self.conv(x)
        return x

class ConvModule(nn.Module):
    def __init__(self, input_dim, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.point_conv1 = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            PointwiseConv(input_dim, 2 * channels, stride=1, bias=True),
            nn.GLU(dim=1)
        )
        self.depth_conv = nn.Sequential(
            DepthwiseConv(channels, channels, kernel_size, stride=1, padding='valid', bias=True),
            nn.BatchNorm1d(channels),
            Swish(),
        )
        self.point_conv2 = nn.Sequential(
            PointwiseConv(channels, input_dim, stride=1, bias=True),
            nn.PReLU()
        )
    def forward(self,x):
        # b,f,t
        x = self.point_conv1(x)
        # should be casual
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.depth_conv(x)
        out = self.point_conv2(x)
        return out

class FFN(nn.Module):
    def __init__(self, input_dim, ffn_dim, gru=False):
        super().__init__()
        self.gru = gru
        self.sequential1 = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.GRU(input_dim, ffn_dim, 1) if gru else torch.nn.Linear(input_dim, ffn_dim, bias=True),
        )
        self.sequential2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, input_dim, bias=True)
        )
    def forward(self,input):
        if self.gru:
            input, _ = self.sequential1(input)
        else:
            input = self.sequential1(input)
        input = self.sequential2(input)
        return input

class TCN(nn.Module):
    '''
    required:    (b,f,t)
    input_dim:  frequency dim or channel dim of input,
    '''
    def __init__(self,input_dim, kernel_size, dilation=1):
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.r = 1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.PReLU(),
            PointwiseConv(self.input_dim, self.input_dim//self.r)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(input_dim//self.r),
            nn.PReLU(),
            DepthwiseConv(self.input_dim//self.r, self.input_dim//self.r, kernel_size,
                        padding='valid',
                        dilation=self.dilation),
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(input_dim//self.r),
            nn.PReLU(),
            PointwiseConv(self.input_dim//self.r, self.input_dim)
        )
    
    def forward(self,x):
        # b,f,t
        residual = x
        x = self.conv1(x)
        # should be casual
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        return x

class Block(nn.Module):
    '''
    required:   (b,t,f)
    input_dim:  frequency dim or channel dim of input,
    ffn_dim:    frequency dim or channel dim inside attention,
    hidden_dim: frequency dim or channel dim inside convolution,
    num_frames: time dim of input,
    '''
    def __init__(self, input_dim, ffn_dim, hidden_dim, kernel_size, num_heads):
        super().__init__()
        self.kernel_size = kernel_size

        self.ffn1 = FFN(input_dim, ffn_dim, gru=False)

        self.time_attn_layer_norm = nn.LayerNorm(input_dim)
        self.time_attention = nn.MultiheadAttention(input_dim,num_heads,batch_first=True)

        self.conv_block = ConvModule(input_dim, hidden_dim, kernel_size)
        
        self.freq_attention_1_avg = nn.AvgPool1d(kernel_size=11, stride=1)
        self.freq_attention_1_max = nn.MaxPool1d(kernel_size=11, stride=1)
        self.freq_attention_2 = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.PReLU(),
            nn.Linear(input_dim//2, input_dim),
        )

        self.ffn2 = FFN(input_dim, ffn_dim, gru=False)

    def generate_square_subsequent_mask(self, sz):
            """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
                Unmasked positions are filled with float(0.0).
            """
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

    def forward(self,input):
        # ffn_1: b,t,f
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        # time attention: b,t,f
        residual = x
        x = self.time_attn_layer_norm(x)
        n_frame = x.shape[1]
        mask = self.generate_square_subsequent_mask(n_frame).cuda()
        x, _ = self.time_attention(query=x,key=x,value=x,need_weights=True,attn_mask=mask)
        x = x + residual

        # conv: b,t,f
        residual = x
        x = x.transpose(-1,-2)   # b,f,t
        x = self.conv_block(x)
        x = x.transpose(-1,-2)
        x = residual + x

        # freq attention: b,t,f
        residual = x
        x = x.transpose(-1, -2) # -> b,f,t
        x = F.pad(x, (11 - 1, 0)) # -> b,f,t+10
        x_avg = self.freq_attention_1_avg(x)    # -> b,f,t
        x_max = self.freq_attention_1_max(x)    # -> b,f,t
        x_avg = x_avg.transpose(-1, -2) # -> b,t,f
        x_max = x_max.transpose(-1, -2) # -> b,t,f
        x_avg = self.freq_attention_2(x_avg)
        x_max = self.freq_attention_2(x_max)
        x = residual * torch.sigmoid(x_avg + x_max)

        # ffn_2: b,t,f
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        return x


class Net(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.w = torch.hann_window(512+2)[1:-1].cuda()

        ######### STAGE I ##############
        self.B = cfg['B']
        self.H = cfg['H']
        self.L = cfg['L']
        self.ffn_dim = cfg['ffn_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.kernel_size = 3
        self.encoder = nn.ModuleList([
            TCN(256, 1) for i in range(self.L)
        ])
        self.net = nn.ModuleList([
            Block(256,
                self.ffn_dim,
                self.hidden_dim,
                self.kernel_size,
                self.H) for i in range(self.B)
        ])
        self.decoder = nn.ModuleList([
            TCN(256, 1) for i in range(self.L)
        ])
        self.last_linear = nn.Linear(256,257)

    def forward(self,input):
        '''
        input:  noisy waveform of (b,t)
        output: irm of (b,f,t),
                enhanced waveform of (b,t')
        '''
        noisy_wav = input

        ########## apply stft, return (b,f,t) ################
        noisy_cmp = stft(noisy_wav,window=self.w,n_fft=512,hop_length=256,
                            center=True,return_complex=True)
        noisy_abs, noisy_ang = torch.abs(noisy_cmp), torch.angle(noisy_cmp)

        # Omit DC opponent
        mag = noisy_abs[:,1:,:]
        residual_list = []
        for layer in self.encoder:
            mag = layer(mag)
            residual_list.append(mag)

        mag = mag.transpose(-1, -2)
        for layer in self.net: 
            mag = layer(mag)
        mag = mag.transpose(-1, -2)     # ->(b,f',t)

        for layer in self.decoder:
            residual = residual_list.pop()
            mag = mag + residual
            mag = layer(mag)
        mag = self.last_linear(mag.transpose(-1, -2)).transpose(-1, -2)     # ->(b,f,t)
        irm = torch.sigmoid(mag)

        ################ recover waveform, return (b,t')##############
        enhanced_cmp = noisy_abs * irm * torch.exp(1j * noisy_ang)
        enhanced_wav = istft(enhanced_cmp,window=self.w,n_fft=512,hop_length=256,center=True)

        return irm, enhanced_wav