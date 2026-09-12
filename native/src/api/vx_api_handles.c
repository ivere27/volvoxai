#include "vx_api_handles.h"

#include "vx_thread.h"
#include "vx_lifecycle.h"
#include <stdlib.h>
#include <string.h>

/* One open-addressing table per handle kind. Ids come from a single global
 * counter so an id is unique across kinds too, which turns a cross-kind mix-up
 * into a lookup miss rather than a silent type confusion. */
typedef struct VxApiSlot {
    int64_t id;      /* 0 empty, -1 tombstone, otherwise the issued id */
    void* pointer;
    VxApiHandleRetainFn retain;
    VxApiHandleReleaseFn release;
    VxApiHandleLineage lineage;
} VxApiSlot;

typedef struct VxApiTable {
    VxApiSlot* slots;
    size_t capacity;  /* always a power of two, or zero before first insert */
    size_t live;
    size_t occupied;  /* live + tombstones, bounds the probe length */
} VxApiTable;

typedef struct VxApiRoot {
    VxRuntime* runtime;
    struct VxApiRoot* previous;
    struct VxApiRoot* next;
} VxApiRoot;

struct VxApiRegistry {
    pthread_mutex_t mutex;
    VxApiTable tables[VX_API_HANDLE_KIND_COUNT];
    VxApiPending* pending;
    VxApiPending* scan;
    VxApiRoot* roots;
    VxApiRoot* root_scan;
    void (*wakeup)(void*);
    void* wakeup_context;
};

static pthread_mutex_t vx_api_issuer_mutex = PTHREAD_MUTEX_INITIALIZER;
/* Keep the issuer unsigned so exhausting the positive int64 namespace fails
 * closed instead of invoking signed-overflow undefined behaviour. */
static uint64_t vx_api_next_id = 1u;
#if defined(__wasm__)
__attribute__((import_module("host"), import_name("vx_host_random_u64_v1")))
extern uint64_t vx_host_random_u64_v1(void);
static int vx_api_owner_seeded;
#endif

#define VX_API_TOMBSTONE ((int64_t)-1)
#define VX_API_MIN_CAPACITY ((size_t)16)

static size_t vx_api_slot_for(const VxApiTable* table, int64_t id) {
    /* Ids are dense and monotonic, so mixing the high bits keeps clustering
     * away from the sequential low bits. */
    uint64_t hash = (uint64_t)id;
    hash ^= hash >> 33;
    hash *= UINT64_C(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    return (size_t)hash & (table->capacity - 1u);
}

/* Reinserts every live entry into a table of the requested capacity. */
static int vx_api_rehash(VxApiTable* table, size_t capacity) {
    VxApiSlot* slots;
    size_t index;

    if (capacity < VX_API_MIN_CAPACITY) capacity = VX_API_MIN_CAPACITY;
    slots = (VxApiSlot*)calloc(capacity, sizeof(*slots));
    if (!slots) return 0;

    if (table->slots) {
        VxApiTable next;
        next.slots = slots;
        next.capacity = capacity;
        for (index = 0; index < table->capacity; index++) {
            const VxApiSlot* slot = &table->slots[index];
            size_t probe;
            if (slot->id <= 0) continue;
            probe = vx_api_slot_for(&next, slot->id);
            while (slots[probe].id > 0) probe = (probe + 1u) & (capacity - 1u);
            slots[probe] = *slot;
        }
        free(table->slots);
    }

    table->slots = slots;
    table->capacity = capacity;
    table->occupied = table->live;
    return 1;
}

static VxApiTable* vx_api_table(VxApiRegistry* registry, VxApiHandleKind kind) {
    if (!registry || kind < 0 || kind >= VX_API_HANDLE_KIND_COUNT) return NULL;
    return &registry->tables[kind];
}

int64_t vx_api_handle_insert(VxApiRegistry* registry, VxApiHandleKind kind,
                             void* pointer,
                             VxApiHandleRetainFn retain,
                             VxApiHandleReleaseFn release) {
    return vx_api_handle_insert_with_lineage(registry, kind, pointer, retain, release, NULL);
}

int64_t vx_api_handle_insert_with_lineage(VxApiRegistry* registry,
    VxApiHandleKind kind,
    void* pointer,
    VxApiHandleRetainFn retain,
    VxApiHandleReleaseFn release,
    const VxApiHandleLineage* parent_lineage) {
    VxApiTable* table;
    VxApiHandleLineage lineage = VX_API_HANDLE_LINEAGE_INIT;
    int64_t id;
    size_t probe;

    if (!pointer || !retain || !release) return 0;
    table = vx_api_table(registry, kind);
    if (!table) return 0;

    pthread_mutex_lock(&registry->mutex);
    pthread_mutex_lock(&vx_api_issuer_mutex);
#if defined(__wasm__)
    if (!vx_api_owner_seeded) {
        /* A worker restart gets an independent opaque handle namespace.
         * Leave half the positive int64 range for monotonic issuance. */
        vx_api_next_id = 1u + (vx_host_random_u64_v1() & (UINT64_MAX >> 2u));
        vx_api_owner_seeded = 1;
    }
#endif
    if (vx_api_next_id > (uint64_t)INT64_MAX) {
        pthread_mutex_unlock(&vx_api_issuer_mutex);
        pthread_mutex_unlock(&registry->mutex);
        return 0;
    }
    id = (int64_t)vx_api_next_id++;
    pthread_mutex_unlock(&vx_api_issuer_mutex);
    /* Grow at half load so probe chains stay short even with tombstones. */
    if (table->capacity == 0u || (table->occupied + 1u) * 2u > table->capacity) {
        size_t target = table->capacity ? table->capacity : VX_API_MIN_CAPACITY;
        while ((table->live + 1u) * 2u > target) target *= 2u;
        if (!vx_api_rehash(table, target)) {
            pthread_mutex_unlock(&registry->mutex);
            return 0;
        }
    }

    if (parent_lineage) lineage = *parent_lineage;
    switch (kind) {
        case VX_API_HANDLE_RUNTIME:
            lineage.runtime_id = (uint64_t)id;
            break;
        case VX_API_HANDLE_MODEL:
            lineage.model_id = (uint64_t)id;
            break;
        case VX_API_HANDLE_COMPILED_MODEL:
            lineage.compiled_model_id = (uint64_t)id;
            break;
        case VX_API_HANDLE_CONTEXT:
            lineage.context_id = (uint64_t)id;
            break;
        case VX_API_HANDLE_GRAPH_PLAN:
            lineage.graph_plan_id = (uint64_t)id;
            break;
        default:
            break;
    }
    probe = vx_api_slot_for(table, id);
    while (table->slots[probe].id > 0) probe = (probe + 1u) & (table->capacity - 1u);
    if (table->slots[probe].id == 0) table->occupied++;
    table->slots[probe].id = id;
    table->slots[probe].pointer = pointer;
    table->slots[probe].retain = retain;
    table->slots[probe].release = release;
    table->slots[probe].lineage = lineage;
    table->live++;
    pthread_mutex_unlock(&registry->mutex);
    return id;
}

/* Locates the live slot for id, or SIZE_MAX. Caller holds the mutex. */
static size_t vx_api_find(const VxApiTable* table, int64_t id) {
    size_t probe;
    size_t scanned;

    if (!table->slots || id <= 0) return (size_t)-1;
    probe = vx_api_slot_for(table, id);
    for (scanned = 0; scanned < table->capacity; scanned++) {
        const VxApiSlot* slot = &table->slots[probe];
        if (slot->id == 0) return (size_t)-1;
        if (slot->id == id) return probe;
        probe = (probe + 1u) & (table->capacity - 1u);
    }
    return (size_t)-1;
}

int vx_api_handle_acquire(VxApiRegistry* registry, VxApiHandleKind kind,
                          int64_t id,
                          VxApiHandleLease* lease) {
    VxApiTable* table = vx_api_table(registry, kind);
    size_t slot;

    if (!table || !lease || lease->pointer || lease->release) return 0;
    pthread_mutex_lock(&registry->mutex);
    slot = vx_api_find(table, id);
    if (slot != (size_t)-1) {
        VxApiSlot* entry = &table->slots[slot];
        entry->retain(entry->pointer);
        lease->pointer = entry->pointer;
        lease->release = entry->release;
        lease->lineage = entry->lineage;
    }
    pthread_mutex_unlock(&registry->mutex);
    return slot != (size_t)-1;
}

void vx_api_handle_lease_release(VxApiHandleLease* lease) {
    void* pointer;
    VxApiHandleReleaseFn release;

    if (!lease || !lease->pointer) return;
    pointer = lease->pointer;
    release = lease->release;
    lease->pointer = NULL;
    lease->release = NULL;
    lease->lineage = (VxApiHandleLineage)VX_API_HANDLE_LINEAGE_INIT;
    release(pointer);
}

int vx_api_handle_remove(VxApiRegistry* registry, VxApiHandleKind kind, int64_t id) {
    VxApiTable* table = vx_api_table(registry, kind);
    void* pointer = NULL;
    VxApiHandleReleaseFn release = NULL;
    size_t slot;

    if (!table) return 0;
    pthread_mutex_lock(&registry->mutex);
    slot = vx_api_find(table, id);
    if (slot != (size_t)-1) {
        pointer = table->slots[slot].pointer;
        release = table->slots[slot].release;
        table->slots[slot].id = VX_API_TOMBSTONE;
        table->slots[slot].pointer = NULL;
        table->slots[slot].retain = NULL;
        table->slots[slot].release = NULL;
        table->slots[slot].lineage =
            (VxApiHandleLineage)VX_API_HANDLE_LINEAGE_INIT;
        table->live--;
    }
    pthread_mutex_unlock(&registry->mutex);
    if (pointer) release(pointer);
    return pointer != NULL;
}

VxApiRegistry* vx_api_registry_create(void) {
    VxApiRegistry* registry = calloc(1, sizeof(*registry));
    if (!registry) return NULL;
    if (pthread_mutex_init(&registry->mutex, NULL) != 0) {
        free(registry);
        return NULL;
    }
    return registry;
}

void vx_api_registry_destroy(VxApiRegistry* registry) {
    if (!registry) return;
    /* Stop queued work before retiring descendants. Cancelling a running
     * request leaves its physical resources alive until completion. */
    VxApiTable* requests = &registry->tables[VX_API_HANDLE_REQUEST];
    for (size_t index = 0; index < requests->capacity; ++index) {
        VxApiSlot* slot = &requests->slots[index];
        if (slot->id > 0) {
            VxReport report = VX_REPORT_INIT;
            (void)vx_request_cancel(slot->pointer, &report);
        }
    }
    /* Roots outlive their public IDs, including ReleaseRuntime followed by
     * descendants and requests whose coordinator still owns physical work. */
    for (VxApiRoot* root = registry->roots; root; root = root->next) {
        VxReport report = VX_REPORT_INIT;
        (void)vx_runtime_close(root->runtime, &report);
    }
    /* Descendants first: releasing a public ancestor never revokes an engine
     * reference still owned by a child. Module calls have already drained. */
    for (int kind = VX_API_HANDLE_KIND_COUNT - 1; kind >= 0; --kind) {
        VxApiTable* table = &registry->tables[kind];
        for (size_t index = 0; index < table->capacity; ++index) {
            VxApiSlot* slot = &table->slots[index];
            if (slot->id > 0) slot->release(slot->pointer);
        }
        free(table->slots);
    }
    while (registry->roots) {
        VxApiRoot* root = registry->roots;
        registry->roots = root->next;
        vx_runtime_release(root->runtime);
        free(root);
    }
    pthread_mutex_destroy(&registry->mutex);
    free(registry);
}

int vx_api_registry_track_runtime(VxApiRegistry* registry, VxRuntime* runtime) {
    VxApiRoot* root = calloc(1, sizeof(*root));
    if (!root) return 0;
    vx_runtime_retain(runtime);
    root->runtime = runtime;
    root->next = registry->roots;
    if (root->next) root->next->previous = root;
    registry->roots = root;
    return 1;
}

void vx_api_registry_reap(VxApiRegistry* registry, uint32_t budget) {
    if (!registry->root_scan) registry->root_scan = registry->roots;
    while (registry->root_scan && budget--) {
        VxApiRoot* root = registry->root_scan;
        registry->root_scan = root->next;
        /* The extra module reference is the last one only after every public
         * object, operation lease and worker-owned request has gone away. */
        if (!vx_runtime_is_unique(root->runtime)) continue;
        if (root->previous) root->previous->next = root->next;
        else registry->roots = root->next;
        if (root->next) root->next->previous = root->previous;
        vx_runtime_release(root->runtime);
        free(root);
    }
}

void vx_api_registry_set_wakeup(VxApiRegistry* registry,
                                void (*wakeup)(void*), void* context) {
    registry->wakeup = wakeup;
    registry->wakeup_context = context;
}

void vx_api_pending_add(VxApiRegistry* registry, VxApiPending* pending) {
    pending->registry = registry;
    pending->next = registry->pending;
    registry->pending = pending;
    atomic_init(&pending->ready, 0);
    vx_api_pending_notify(pending);
}

void vx_api_pending_remove(VxApiPending* pending) {
    VxApiRegistry* registry = pending->registry;
    if (!registry) return;
    VxApiPending** entry = &registry->pending;
    while (*entry && *entry != pending) entry = &(*entry)->next;
    if (*entry) *entry = pending->next;
    if (registry->scan == pending) registry->scan = pending->next;
    pending->next = NULL;
    pending->registry = NULL;
}

void vx_api_pending_notify(void* context) {
    VxApiPending* pending = context;
    VxApiRegistry* registry = pending->registry;
    atomic_store_explicit(&pending->ready, 1, memory_order_release);
    if (registry && registry->wakeup) registry->wakeup(registry->wakeup_context);
}

uint32_t vx_api_registry_poll(VxApiRegistry* registry, uint32_t budget) {
    uint32_t count = 0;
    /* A host notification may represent a GPU completion, which becomes
     * visible to C only when it inspects its live ticket. Scan waiting calls
     * once per notification and retain the cursor across bounded turns. */
    if (!registry->scan) registry->scan = registry->pending;
    while (registry->scan && count < budget) {
        VxApiPending* pending = registry->scan;
        registry->scan = pending->next;
        atomic_store_explicit(&pending->ready, 0, memory_order_release);
        pending->poll(pending);
        ++count;
    }
    return count;
}

int vx_api_registry_has_work(VxApiRegistry* registry) {
    if (registry->scan) return 1;
    for (VxApiPending* pending = registry->pending; pending; pending = pending->next)
        if (atomic_load_explicit(&pending->ready, memory_order_acquire)) return 1;
    return 0;
}
