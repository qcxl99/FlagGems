# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# These two kernels used to live in this backend's `_conj.py`, which has been
# removed along with the rest of the Gems `_conj` operator. `conj_physical`
# stays, so the stride-1 storage walk it relies on is kept here: a strided
# (stride-2) pass over complex storage is discrete access on XPU and was
# measured at a 0.007-0.040 speedup against eager, so falling back to the
# generic implementation would be a large regression on this backend.
@triton.jit
def _conj_flat_kernel(fin, fout, n2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n2
    x = tl.load(fin + i, mask=m)
    out = tl.where((i % 2) == 1, -x, x)
    tl.store(fout + i, out, mask=m)


@triton.jit
def _conj_flat_copy_kernel(fin, fout, n2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n2
    x = tl.load(fin + i, mask=m)
    tl.store(fout + i, x, mask=m)


def _flatten_storage(input: torch.Tensor) -> torch.Tensor:
    """Return a contiguous 1D view of the interleaved real/imag storage.

    ``view_as_real`` strips the complex dtype, so materializing a
    non-contiguous input goes through the vendor ``copy_`` for a *real* tensor
    (the complex ``copy_`` is rejected by the vendor on this stack).
    """
    rv = torch.view_as_real(input)
    if not rv.is_contiguous():
        rv = rv.contiguous()
    return rv.reshape(-1)


def _conj_block_size(n2: int) -> int:
    return 32768 if n2 >= 1 << 20 else 8192


def conj_physical(input: torch.Tensor) -> torch.Tensor:
    """Materialize the complex conjugate with the vendor stride-1 kernel."""
    logger.debug("GEMS_KUNLUNXIN CONJ_PHYSICAL")
    if not input.is_complex():
        return input

    out = torch.empty_like(input, memory_format=torch.contiguous_format)
    fin = _flatten_storage(input)
    fout = torch.view_as_real(out).reshape(-1)
    n2 = fin.numel()

    BLOCK = _conj_block_size(n2)
    grid = (triton.cdiv(n2, BLOCK),)
    with torch_device_fn.device(input.device):
        # triton reads raw storage and ignores the lazy conjugate bit. When the
        # bit is set the stored value is already the conjugate of the logical
        # value, so copying the storage verbatim *is* the physical conjugate;
        # otherwise the imaginary lanes are negated. This folds the previous
        # resolve_conj() + _conj() pair into a single pass.
        if input.is_conj():
            _conj_flat_copy_kernel[grid](fin, fout, n2, BLOCK=BLOCK, num_warps=8)
        else:
            _conj_flat_kernel[grid](fin, fout, n2, BLOCK=BLOCK, num_warps=8)

    return out
