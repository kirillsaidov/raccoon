#include "raccoon/auxiliary/tape.h"

/* 
    Tape creation/destruction
*/

rac_tape_t *rac_tape_make(struct VitaBaseAllocatorType *const alloctr) {
    // allocate for variable
    rac_tape_t *tape = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_tape_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_tape_t));
    
    // init
    *tape = (rac_tape_t) {
        .list = vt_plist_create(VT_ARRAY_DEFAULT_INIT_ELEMENTS, alloctr),
        .alloctr = alloctr,
    };

    return tape;
}

void rac_tape_free(rac_tape_t *tape) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // free list elements
    rac_var_t *tmp = NULL;
    while ((tmp = vt_plist_pop_get(tape->list)) != NULL) {
        rac_var_free(tmp);
    }

    // free list
    vt_plist_destroy(tape->list);

    // free tape
    (tape->alloctr) ? VT_ALLOCATOR_FREE(tape->alloctr, tape) : VT_FREE(tape);
}

/* 
    Tape operations
*/

void rac_tape_reset(rac_tape_t *const tape) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    
    // free list elements
    rac_var_t *tmp = NULL;
    while ((tmp = vt_plist_pop_get(tape->list)) != NULL) {
        rac_var_free(tmp);
    }

    // unlock
    tape->locked = false;
}

void rac_tape_update(rac_tape_t *const tape) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // update
    rac_var_t *tmp = NULL;
    while ((tmp = vt_plist_slide_front(tape->list)) != NULL) {
        rac_var_update(tmp);
    }
}

void rac_tape_push(rac_tape_t *const tape, const rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(!tape->locked, "%s\n", "Cannot push to the tape! Need to `rac_tape_reset(tape)` first!");
    vt_plist_push_back(tape->list, var);
}

void rac_tape_push_ex(rac_tape_t *const tape, const size_t arr_size, rac_var_t *arr[arr_size]) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(arr != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(arr_size > 0, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(!tape->locked, "%s\n", "Cannot push to the tape! Need to `rac_tape_reset(tape)` first!");

    // push elements
    VT_FOREACH(i, 0, arr_size) {
        VT_ENFORCE(arr[i] != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
        vt_plist_push_back(tape->list, arr[i]);
    }
}

rac_var_t *rac_tape_first(const rac_tape_t *const tape) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return vt_plist_len(tape->list) 
        ? vt_plist_get(tape->list, 0) 
        : NULL;
}

rac_var_t *rac_tape_get(const rac_tape_t *const tape, const size_t idx) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(
        idx < vt_plist_len(tape->list),
        "%s: Out of bounds memory access at %zu, but length is %zu!\n", 
        vt_status_to_str(VT_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS), 
        idx, 
        vt_plist_len(tape->list)
    );

    return vt_plist_get(tape->list, idx);
}

rac_var_t *rac_tape_last(const rac_tape_t *const tape) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return vt_plist_len(tape->list) 
        ? vt_plist_get(tape->list, vt_plist_len(tape->list)-1) 
        : NULL;
}

void rac_tape_compile(rac_tape_t *const tape) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    tape->locked = true;
}

bool rac_tape_compiled(const rac_tape_t *const tape) {
    // check for invalid input
    VT_DEBUG_ASSERT(tape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return tape->locked;
}

