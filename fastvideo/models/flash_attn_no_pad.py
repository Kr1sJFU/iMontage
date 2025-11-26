'''
For Flash Attn 3 verision, you should replace the commented code with others
'''

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# from flash_attn_interface import flash_attn_varlen_func 
from flash_attn.bert_padding import pad_input, unpad_input
from einops import rearrange

'''
Note we train and eval on Flash Attention
'''
# def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None):
#     """
#     qkv: Tensor of shape [B, S, 3, H, D]
#     key_padding_mask: BoolTensor of shape [B, S]
#     """
#     B, S, three, H, D = qkv.shape

#     x = rearrange(qkv, "b s three h d -> b s (three h d)")

#     #    x_unpad:    [NNZ, three * H * D]
#     #    indices:    用于还原回原始位置
#     #    cu_seqlens: [B+1] prefix sum，用来定位每个样本在 x_unpad 里的起止
#     #    max_s:      去 pad 后最长序列长度
#     x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x, key_padding_mask)

#     x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=H)
#     q_unpad = x_unpad[:, 0]  # [NNZ, H, D]
#     k_unpad = x_unpad[:, 1]  # [NNZ, H, D]
#     v_unpad = x_unpad[:, 2]  # [NNZ, H, D]

#     output_unpad, _ = flash_attn_varlen_func(
#         q_unpad,
#         k_unpad,
#         v_unpad,
#         cu_seqlens,   # cu_seqlens_q
#         cu_seqlens,   # cu_seqlens_k
#         max_s,        # max_seqlen_q
#         max_s,        # max_seqlen_k
#         softmax_scale=softmax_scale,
#         causal=causal,
#     ) 

#     out_flat = rearrange(output_unpad, "nnz h d -> nnz (h d)")
#     padded = pad_input(out_flat, indices, B, S)  # [B, S, H*D]

#     output = rearrange(padded, "b s (h d) -> b s h d", h=H)
#     return output

def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None):
    """
    qkv: Tensor, [B, S, 3, H, D]
    key_padding_mask: BoolTensor, [B, S]
    """
    B, S, three, H, D = qkv.shape

    x = rearrange(qkv, "b s three h d -> b s (three h d)")

    x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x, key_padding_mask)

    qkv_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=H)

    output_unpad = flash_attn_varlen_qkvpacked_func(
        qkv_unpad,
        cu_seqlens,
        max_s,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    
    out_flat = rearrange(output_unpad, "nnz h d -> nnz (h d)")  # [NNZ, H*D]
    padded = pad_input(out_flat, indices, B, S)                 # [B, S, H*D]

    output = rearrange(padded, "b s (h d) -> b s h d", h=H)
    return output
