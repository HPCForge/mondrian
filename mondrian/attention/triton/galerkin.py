r"""
This is a modification of the Flash-Attention-2 implemented in the triton tutorial.
This is only a non-causal form.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _galerkin_forward_kernel(
    query_ptr,
    value_ptr,
    key_ptr,
    out_ptr,
    # q is mxk,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    # k is nxk
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    # v is nxp
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vp,
    # o is mxp
    stride_oz,
    stride_oh,
    stride_om,
    stride_op,
    batch_size,
    num_heads,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // num_heads
    off_h = off_hz % num_heads
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    query_block_ptr = tl.make_block_ptr(
        base=query_ptr + qkv_offset,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # key blocks are transposed
    key_block_ptr = tl.make_block_ptr(
        base=key_ptr + qkv_offset,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    value_block_ptr = tl.make_block_ptr(
        base=value_ptr + qkv_offset,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_vn, stride_vp),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + qkv_offset,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_om, stride_op),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    acc_ktv = tl.zeros([HEAD_DIM, HEAD_DIM], dtype=tl.float32)
    tl.static_print(acc_ktv)
    for start_n in range(0, seq_len, BLOCK_N):
        k = tl.load(key_block_ptr)
        v = tl.load(value_block_ptr)
        acc_ktv += tl.dot(tl.trans(k), v)  # acc=acc_ktv)
        key_block_ptr = tl.advance(key_block_ptr, (BLOCK_N, 0))
        value_block_ptr = tl.advance(value_block_ptr, (BLOCK_N, 0))

    tl.device_print("acc_ktv", acc_ktv)

    query = tl.load(query_block_ptr)

    acc = tl.dot(query, acc_ktv)

    tl.device_print("acc", acc)

    tl.store(out_block_ptr, acc.to(out_ptr.type.element_ty))


class _galerkin_attention(torch.autograd.Function):
    def forward(ctx, query, key, value):
        out = torch.zeros_like(query)

        grid = lambda args: (
            triton.cdiv(query.size(2), args["BLOCK_M"]),
            query.size(0) * query.size(1),
            1,
        )
        _galerkin_forward_kernel[grid](
            query,
            key,
            value,
            out,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            batch_size=query.size(0),
            num_heads=query.size(1),
            seq_len=query.size(2),
            HEAD_DIM=query.size(3),
            BLOCK_M=16,
            BLOCK_N=16,
        )

        ctx.save_for_backward(query, key, value, out)
        ctx.grid = grid
        ctx.head_dim = query.size(3)
        return out

    def backward(ctx, dout):
        pass


galerkin_attention = _galerkin_attention.apply
