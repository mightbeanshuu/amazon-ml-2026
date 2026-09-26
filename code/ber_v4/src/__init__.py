import os

# pyarrow's default (jemalloc/mimalloc) pool keeps freed memory; on an 8 GB machine that shows up as steadily
# growing swap. The system allocator returns memory to the OS. Must be set before pyarrow is imported.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# sparse_dot_topn and LightGBM each bring an OpenMP runtime. If LightGBM's libomp is loaded first, a
# multi-threaded sp_matmul_topn call segfaults (reproduced on macOS arm64). Importing sparse_dot_topn
# first makes both work, so it is done here, before any submodule.
import sparse_dot_topn  # noqa: F401
