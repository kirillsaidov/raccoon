#include "raccoon/variable.h"

static vt_plist_t *rac_var_build_parent_tree(rac_var_t *const node_start);
static void rac_var_deep_walk(rac_var_t *const node_curr, vt_plist_t *const node_list);
static void rac_var_add_backward(rac_var_t *const op_result);
static void rac_var_mul_backward(rac_var_t *const op_result);

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

    // build parent tree
    vt_plist_t *node_list = rac_var_build_parent_tree(var);

    // base case
    var->grad = 1;

    // zero out gradients
    const size_t len = vt_plist_len(node_list);
    VT_FOREACH(i, 0, len) {
        rac_var_t *node = vt_plist_get(node_list, i);
        if (node->backward) node->backward(node);
    }

    // free parent tree
    vt_plist_destroy(node_list);
}

void rac_var_zero_grad(rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // build parent tree
    vt_plist_t *node_list = rac_var_build_parent_tree(var);

    // zero out gradients
    const size_t len = vt_plist_len(node_list);
    VT_FOREACH(i, 0, len) {
        rac_var_t *node = vt_plist_get(node_list, i);
        node->grad = 0;
    }

    // free parent tree
    vt_plist_destroy(node_list);
}

/* 
    Variable operations
*/

rac_var_t *rac_var_add(rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return rac_var_make_ex(lhs->alloctr, lhs->data + rhs->data, (rac_var_t*[2]){lhs, rhs}, rac_var_add_backward);
}

rac_var_t *rac_var_sub(rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return rac_var_make_ex(lhs->alloctr, lhs->data - rhs->data, (rac_var_t*[2]){lhs, rhs}, rac_var_add_backward);
}

rac_var_t *rac_var_mul(rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return rac_var_make_ex(lhs->alloctr, lhs->data * rhs->data, (rac_var_t*[2]){lhs, rhs}, rac_var_mul_backward);
}

rac_var_t *rac_var_div(rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return rac_var_make_ex(lhs->alloctr, lhs->data / rhs->data, (rac_var_t*[2]){lhs, rhs}, rac_var_mul_backward);
}

// -------------------------- PRIVATE -------------------------- //

static vt_plist_t *rac_var_build_parent_tree(rac_var_t *const node_start) {
    // check for invalid input
    VT_DEBUG_ASSERT(node_start != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // create node list
    vt_plist_t *node_list = vt_plist_create(VT_ARRAY_DEFAULT_INIT_ELEMENTS, node_start->alloctr);

    // walk all tree nodes
    rac_var_deep_walk(node_start, node_list);

    return node_list;
}

static void rac_var_deep_walk(rac_var_t *const node_curr, vt_plist_t *const node_list) {
    // check for invalid input
    VT_DEBUG_ASSERT(node_list != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // if current node NULL, return
    if (node_curr == NULL) return;

    // add node to node list
    if (vt_plist_can_find(node_list, node_curr) < 0) vt_plist_push_back(node_list, node_curr);

    // traverse each parent node iteratively
    VT_FOREACH(i, 0, RAC_VAR_PARENTS_LEN) rac_var_deep_walk(node_curr->parents[i], node_list);
}

static void rac_var_add_backward(rac_var_t *const op_result) {
    // check for invalid input
    VT_DEBUG_ASSERT(op_result != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // get lhs, rhs
    rac_var_t *const lhs = op_result->parents[0];
    rac_var_t *const rhs = op_result->parents[1];

    // perform backward operation
    lhs->grad += 1.0 * op_result->grad;
    rhs->grad += 1.0 * op_result->grad;
}

static void rac_var_mul_backward(rac_var_t *const op_result) {
    // check for invalid input
    VT_DEBUG_ASSERT(op_result != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // get lhs, rhs
    rac_var_t *const lhs = op_result->parents[0];
    rac_var_t *const rhs = op_result->parents[1];

    // perform backward operation
    lhs->grad += rhs->data * op_result->grad;
    rhs->grad += lhs->data * op_result->grad;
}

