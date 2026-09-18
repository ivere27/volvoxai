/* Python capsule ABI adaptation only. Every engine operation still goes
 * through the generated protobuf client. No framework or CUDA SDK linkage. */
#define Py_LIMITED_API 0x030A0000
#include <Python.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include "dlpack.h"

static atomic_int shutting_down;

typedef struct {
    void* managed;
    int versioned;
} ImportOwner;

static void imported_destroy(PyObject* capsule) {
    ImportOwner* owner = PyCapsule_GetPointer(capsule, "volvoxai.dlpack.owner");
    if (!owner) { PyErr_Clear(); return; }
    if (!shutting_down) {
        if (owner->versioned) {
            DLManagedTensorVersioned* tensor = owner->managed;
            if (tensor->deleter) tensor->deleter(tensor);
        } else {
            DLManagedTensor* tensor = owner->managed;
            if (tensor->deleter) tensor->deleter(tensor);
        }
    }
    free(owner);
}

static PyObject* consume(PyObject* self, PyObject* capsule) {
    (void)self;
    int versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    const char* name = versioned ? "dltensor_versioned" : "dltensor";
    if (!versioned && !PyCapsule_IsValid(capsule, name)) {
        PyErr_SetString(PyExc_ValueError, "DLPack capsule is invalid or already consumed");
        return NULL;
    }
    void* managed = PyCapsule_GetPointer(capsule, name);
    if (!managed) return NULL;
    DLTensor* tensor;
    if (versioned) {
        DLManagedTensorVersioned* value = managed;
        if (value->version.major != DLPACK_MAJOR_VERSION) {
            PyErr_SetString(PyExc_BufferError, "Unsupported DLPack major version");
            return NULL;
        }
        tensor = &value->dl_tensor;
    } else tensor = &((DLManagedTensor*)managed)->dl_tensor;
    if (tensor->ndim < 0 || tensor->ndim > 8 || (tensor->ndim && !tensor->shape)) {
        PyErr_SetString(PyExc_BufferError, "DLPack rank must be between zero and eight");
        return NULL;
    }
    ImportOwner* owner = calloc(1, sizeof(*owner));
    if (!owner) return PyErr_NoMemory();
    owner->managed = managed;
    owner->versioned = versioned;
    PyObject* lease = PyCapsule_New(owner, "volvoxai.dlpack.owner", imported_destroy);
    if (!lease) { free(owner); return NULL; }
    if (PyCapsule_SetName(capsule, versioned ? "used_dltensor_versioned" : "used_dltensor") < 0) {
        /* Ownership has not transferred; the original capsule still owns it. */
        PyCapsule_SetDestructor(lease, NULL);
        free(owner);
        Py_DECREF(lease);
        return NULL;
    }
    PyObject* shape = PyTuple_New(tensor->ndim);
    PyObject* strides = tensor->strides ? PyTuple_New(tensor->ndim) : Py_NewRef(Py_None);
    if (!shape || !strides) { Py_XDECREF(shape); Py_XDECREF(strides); Py_DECREF(lease); return NULL; }
    for (int i = 0; i < tensor->ndim; i++) {
        PyObject* extent = PyLong_FromLongLong(tensor->shape[i]);
        if (!extent || PyTuple_SetItem(shape, i, extent) < 0) goto fail;
        if (tensor->strides) {
            PyObject* stride = PyLong_FromLongLong(tensor->strides[i]);
            if (!stride || PyTuple_SetItem(strides, i, stride) < 0) goto fail;
        }
    }
    PyObject* descriptor = Py_BuildValue("{s:K,s:K,s:i,s:i,s:i,s:i,s:i,s:O,s:O}",
        "address", (unsigned long long)(uintptr_t)tensor->data,
        "offset", (unsigned long long)tensor->byte_offset,
        "device_type", (int)tensor->device.device_type, "device_id", tensor->device.device_id,
        "code", (int)tensor->dtype.code, "bits", (int)tensor->dtype.bits,
        "lanes", (int)tensor->dtype.lanes, "shape", shape, "strides", strides);
    Py_DECREF(shape); Py_DECREF(strides);
    if (!descriptor) { Py_DECREF(lease); return NULL; }
    return Py_BuildValue("NN", descriptor, lease);
fail:
    Py_DECREF(shape); Py_DECREF(strides); Py_DECREF(lease);
    return NULL;
}

typedef struct Export {
    DLManagedTensor legacy;
    DLManagedTensorVersioned versioned;
    int64_t shape[8];
    int64_t strides[8];
    PyObject* owner;
    void* native_managed;
    int native_versioned;
    struct Export* next;
} Export;

static _Atomic(Export*) pending_exports;

static int drain_exports(void* unused) {
    (void)unused;
    PyObject *error_type, *error_value, *error_traceback;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    Export* value = atomic_exchange(&pending_exports, NULL);
    while (value) {
        Export* next = value->next;
        Py_DECREF(value->owner);
        free(value);
        value = next;
    }
    PyErr_Restore(error_type, error_value, error_traceback);
    return 0;
}

static void export_destroy(Export* value) {
    /* Framework deleters can run without a Python thread state. Defer Python
     * decrefs to a pending interpreter callback instead of entering the GIL
     * from a foreign thread that may race interpreter shutdown. */
    if (atomic_load(&shutting_down) || !Py_IsInitialized()) {
        free(value);
        return;
    }
    if (value->native_managed) {
        if (value->native_versioned) {
            DLManagedTensorVersioned* native = value->native_managed;
            if (native->deleter) native->deleter(native);
        } else {
            DLManagedTensor* native = value->native_managed;
            if (native->deleter) native->deleter(native);
        }
    }
    Export* head = atomic_load(&pending_exports);
    do { value->next = head; }
    while (!atomic_compare_exchange_weak(&pending_exports, &head, value));
    if (!atomic_load(&shutting_down)) (void)Py_AddPendingCall(drain_exports, NULL);
}
static void legacy_destroy(DLManagedTensor* tensor) { export_destroy(tensor->manager_ctx); }
static void versioned_destroy(DLManagedTensorVersioned* tensor) { export_destroy(tensor->manager_ctx); }
static void exported_capsule_destroy(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        DLManagedTensor* tensor = PyCapsule_GetPointer(capsule, "dltensor");
        tensor->deleter(tensor);
    } else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        DLManagedTensorVersioned* tensor = PyCapsule_GetPointer(capsule, "dltensor_versioned");
        tensor->deleter(tensor);
    }
}

static PyObject* make_capsule(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long address, offset;
    int device_type, device_id, code, bits, lanes, minor;
    PyObject *shape, *owner;
    if (!PyArg_ParseTuple(args, "KiiiiiOOiK", &address, &device_type, &device_id,
            &code, &bits, &lanes, &shape, &owner, &minor, &offset)) return NULL;
    Py_ssize_t rank = PyTuple_Size(shape);
    if (rank < 0) return NULL;
    if (rank > 8 || !address) {
        PyErr_SetString(PyExc_ValueError, "Invalid native tensor descriptor");
        return NULL;
    }
    Export* value = calloc(1, sizeof(*value));
    if (!value) return PyErr_NoMemory();
    int64_t stride = 1;
    for (Py_ssize_t i = rank; i-- > 0;) {
        int64_t extent = PyLong_AsLongLong(PyTuple_GetItem(shape, i));
        if (PyErr_Occurred() || extent <= 0 || extent > INT64_MAX / stride) {
            free(value);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "Invalid tensor extent");
            return NULL;
        }
        value->shape[i] = extent;
        value->strides[i] = stride;
        stride *= extent;
    }
    DLTensor tensor = {(void*)(uintptr_t)address, {(DLDeviceType)device_type, device_id},
        (int32_t)rank, {(uint8_t)code, (uint8_t)bits, (uint16_t)lanes},
        value->shape, value->strides, offset};
    value->owner = Py_NewRef(owner);
    PyObject* capsule;
    if (minor >= 0) {
        value->versioned.version = (DLPackVersion){DLPACK_MAJOR_VERSION, (uint32_t)minor};
        value->versioned.manager_ctx = value;
        value->versioned.deleter = versioned_destroy;
        value->versioned.dl_tensor = tensor;
        capsule = PyCapsule_New(&value->versioned, "dltensor_versioned", exported_capsule_destroy);
    } else {
        value->legacy.dl_tensor = tensor;
        value->legacy.manager_ctx = value;
        value->legacy.deleter = legacy_destroy;
        capsule = PyCapsule_New(&value->legacy, "dltensor", exported_capsule_destroy);
    }
    if (!capsule) export_destroy(value);
    return capsule;
}

static PyObject* wrap_managed(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long address;
    int versioned;
    PyObject* owner;
    if (!PyArg_ParseTuple(args, "KpO", &address, &versioned, &owner)) return NULL;
    Export* value = calloc(1, sizeof(*value));
    if (!value) {
        /* This adapter consumes the C export, including allocation failure. */
        if (versioned) {
            DLManagedTensorVersioned* native = (void*)(uintptr_t)address;
            if (native->deleter) native->deleter(native);
        } else {
            DLManagedTensor* native = (void*)(uintptr_t)address;
            if (native->deleter) native->deleter(native);
        }
        return PyErr_NoMemory();
    }
    value->owner = Py_NewRef(owner);
    value->native_managed = (void*)(uintptr_t)address;
    value->native_versioned = versioned;
    PyObject* capsule;
    if (versioned) {
        value->versioned = *(DLManagedTensorVersioned*)value->native_managed;
        value->versioned.manager_ctx = value;
        value->versioned.deleter = versioned_destroy;
        capsule = PyCapsule_New(&value->versioned, "dltensor_versioned", exported_capsule_destroy);
    } else {
        value->legacy = *(DLManagedTensor*)value->native_managed;
        value->legacy.manager_ctx = value;
        value->legacy.deleter = legacy_destroy;
        capsule = PyCapsule_New(&value->legacy, "dltensor", exported_capsule_destroy);
    }
    if (!capsule) export_destroy(value);
    return capsule;
}
static PyObject* capsule_pointer(PyObject* self, PyObject* capsule) {
    (void)self;
    int versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    void* pointer = PyCapsule_GetPointer(capsule, versioned ? "dltensor_versioned" : "dltensor");
    if (!pointer) return NULL;
    return Py_BuildValue("Ki", (unsigned long long)(uintptr_t)pointer, versioned);
}
static PyObject* disown(PyObject* self, PyObject* capsule) {
    (void)self;
    int versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    if (!versioned && !PyCapsule_IsValid(capsule, "dltensor")) {
        PyErr_SetString(PyExc_ValueError, "DLPack capsule is already consumed"); return NULL;
    }
    if (PyCapsule_SetName(capsule, versioned ? "used_dltensor_versioned" : "used_dltensor") < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject* shutdown_bridge(PyObject* self, PyObject* unused) {
    (void)self; (void)unused;
    atomic_store(&shutting_down, 1);
    drain_exports(NULL);
    Py_RETURN_NONE;
}
static PyMethodDef methods[] = {
    {"wrap_managed", wrap_managed, METH_VARARGS, "Wrap a C-owned standard DLPack export; retain module code."},
    {"capsule_pointer", capsule_pointer, METH_O, "Inspect a capsule for a generated native import."},
    {"disown", disown, METH_O, "Mark a successfully imported capsule consumed."},
    {"consume", consume, METH_O, "Consume a DLPack capsule once and retain its producer."},
    {"make_capsule", make_capsule, METH_VARARGS, "Export a declared native buffer with a lifetime owner."},
    {"_shutdown", shutdown_bridge, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_dlpack", NULL, -1, methods};
PyMODINIT_FUNC PyInit__dlpack(void) { return PyModule_Create(&module); }
