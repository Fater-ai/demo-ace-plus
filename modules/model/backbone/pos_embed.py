import numpy as np
from einops import rearrange

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def frame_pad(x, seq_len, shapes):
    max_h, max_w = np.max(shapes, 0)
    frames = []
    cur_len = 0
    for h, w in shapes:
        frame_len = h * w
        frames.append(
            F.pad(
                x[cur_len:cur_len + frame_len].view(h, w, -1),
                (0, 0, 0, max_w - w, 0, max_h - h))  # .view(max_h * max_w, -1)
        )
        cur_len += frame_len
        if cur_len >= seq_len:
            break
    return torch.stack(frames)


def frame_unpad(x, shapes):
    max_h, max_w = np.max(shapes, 0)
    x = rearrange(x, '(b h w) n c -> b h w n c', h=max_h, w=max_w)
    frames = []
    for i, (h, w) in enumerate(shapes):
        if i >= len(x):
            break
        frames.append(rearrange(x[i, :h, :w], 'h w n c -> (h w) n c'))
    return torch.concat(frames)


@amp.autocast(enabled=False)
def rope_apply_multires(x, x_lens, x_shapes, freqs, pad=True):
    """
    x:          [B*L, N, C].
    x_lens:     [B].
    x_shapes:   [B, F, 2].
    freqs:      [M, C // 2].
    """
    n, c = x.size(1), x.size(2) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # loop over samples
    output = []
    st = 0
    for i, (seq_len,
            shapes) in enumerate(zip(x_lens.tolist(), x_shapes.tolist())):
        x_i = frame_pad(x[st:st + seq_len], seq_len, shapes)  # f, h, w, c
        f, h, w = x_i.shape[:3]
        pad_seq_len = f * h * w
        # precompute multipliers
        x_i = torch.view_as_complex(
            x_i.to(torch.float64).reshape(pad_seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(pad_seq_len, 1, -1)
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2).type_as(x)
        x_i = frame_unpad(x_i, shapes)
        # append to collection
        output.append(x_i)
        st += seq_len
    return pad_sequence(output) if pad else torch.concat(output)


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """
    Precompute the frequency tensor for complex exponentials.
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs