"""Explicit owner scope for tests calling private production CPU kernels."""
import ctypes


class _Scope(ctypes.Structure):
    _fields_ = [("previous", ctypes.c_void_p), ("bound", ctypes.c_void_p),
                ("retained", ctypes.c_int)]


def bind_kernel_pool(library, threads=1):
    """Bind until the returned cleanup runs, on this same calling thread."""
    library.vx_kernel_thread_pool_create.argtypes = [ctypes.c_int]
    library.vx_kernel_thread_pool_create.restype = ctypes.c_void_p
    library.vx_kernel_thread_pool_scope_enter.argtypes = [ctypes.c_void_p]
    library.vx_kernel_thread_pool_scope_enter.restype = _Scope
    library.vx_kernel_thread_pool_scope_leave.argtypes = [_Scope]
    library.vx_kernel_thread_pool_destroy.argtypes = [ctypes.c_void_p]
    pool = library.vx_kernel_thread_pool_create(threads)
    if not pool:
        raise MemoryError("production kernel pool allocation failed")
    scope = library.vx_kernel_thread_pool_scope_enter(pool)

    def cleanup():
        library.vx_kernel_thread_pool_scope_leave(scope)
        library.vx_kernel_thread_pool_destroy(pool)

    return cleanup
