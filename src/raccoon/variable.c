#include "raccoon/variable.h"

/* 
    Variable creation/destruction
*/

rac_var_t *rac_var_make(struct VitaBaseAllocatorType *const alloctr, const rac_float data) {
    // allocate for variable
    rac_var_t *var = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_var_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_var_t));

    // init
    *var = (rac_var_t) {
        .data = data,
        .grad = 0,
        .alloctr = alloctr,
    };

    return var;
}

rac_var_t *rac_var_make_ex(struct VitaBaseAllocatorType *const alloctr, const rac_float data, struct RaccoonVariable *parents[2], void (*backward)(struct RaccoonVariable*)) {
    // allocate for variable
    rac_var_t *var = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_var_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_var_t));

    // init
    *var = (rac_var_t) {
        .data = data,
        .grad = 0,
        .parents = { parents[0], parents[1] },
        .backward = backward,
        .alloctr = alloctr,
    };

    return var;
}

void rac_var_free(rac_var_t *var) {
    // check for invalid input
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // free variable
    (var->alloctr) ? VT_ALLOCATOR_FREE(var->alloctr, var) : VT_FREE(var);
}

/* 
    Variable functionality
*/

void rac_var_backward(rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
}

void rac_var_zero_grad(rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
}

/* 
    Variable operations
*/

rac_var_t *rac_var_add(rac_var_t *const var);
rac_var_t *rac_var_sub(rac_var_t *const var);
rac_var_t *rac_var_mul(rac_var_t *const var);
rac_var_t *rac_var_div(rac_var_t *const var);

// -------------------------- PRIVATE -------------------------- //

vt_plist_t *rac_var_build_parent_tree(rac_var_t *const node_start) {
    // check for invalid input
    VT_DEBUG_ASSERT(node_start != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // create node list
    vt_plist_t *node_list = vt_plist_create(VT_ARRAY_DEFAULT_INIT_ELEMENTS, node_start->alloctr);

    // walk all tree nodes
    rac_var_deep_walk(node_start, node_list);

    return node_list;
}

void rac_var_deep_walk(rac_var_t *const node_curr, vt_plist_t *const node_list) {
    // check for invalid input
    VT_DEBUG_ASSERT(node_curr != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(node_list != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // add node to node list
    // if (vt_plist_can_find(node_list, node_curr) < 0) vt_plist_push()
}

