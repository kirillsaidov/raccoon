#include "raccoon/core/variable.h"
#include "vita/math/math.h"

static void rac_var_deep_walk(rac_var_t *const node_curr, vt_plist_t *const node_list);
static void rac_var_add_backward(rac_var_t *const op_result);
static void rac_var_mul_backward(rac_var_t *const op_result);

/* 
    Variable creation/destruction
*/

rac_var_t *rac_var_make(struct VitaBaseAllocatorType *const alloctr, const rac_float data) {
    return rac_var_make_ex(alloctr, data, 0, (rac_var_t*[2]){NULL, NULL}, NULL);
}

rac_var_t *rac_var_make_ex(struct VitaBaseAllocatorType *const alloctr, const rac_float data, const char op, struct RaccoonVariable *parents[2], void (*backward)(struct RaccoonVariable*)) {
    // allocate for variable
    rac_var_t *var = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_var_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_var_t));

    // init
    *var = (rac_var_t) {
        .data = data,
        .grad = 0,
        .op = op,
        .parents = { parents[0], parents[1] },
        .backward = backward,
        .alloctr = alloctr,
    };

    return var;
}

rac_var_t *rac_var_make_rand(struct VitaBaseAllocatorType *const alloctr) {
    // allocate for variable
    rac_var_t *var = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_var_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_var_t));

    // init
    *var = (rac_var_t) {
        .data = vt_math_random_f32_uniform(0, 1),
        .grad = 0,
        .alloctr = alloctr,
    };

    return var;
}

void rac_var_remake(rac_var_t *var, const rac_float data, const char op, struct RaccoonVariable *parents[2], void (*backward)(struct RaccoonVariable*)) {
    // check for invalid input
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // update values
    var->grad = 0;
    var->data = data;
    var->op = op;
    var->parents[0] = parents ? parents[0] : NULL;
    var->parents[1] = parents ? parents[1] : NULL;
    var->backward = backward;
}

void rac_var_free(rac_var_t *var) {
    // check for invalid input
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // free variable
    (var->alloctr) ? VT_ALLOCATOR_FREE(var->alloctr, var) : VT_FREE(var);
}

/* 
    Variable operations
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

    // zero grad
    var->grad = 0;
}

rac_var_t *rac_var_add(rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return rac_var_make_ex(lhs->alloctr, lhs->data + rhs->data, '+', (rac_var_t*[2]){lhs, rhs}, rac_var_add_backward);
}

rac_var_t *rac_var_sub(rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return rac_var_make_ex(lhs->alloctr, lhs->data - rhs->data, '-', (rac_var_t*[2]){lhs, rhs}, rac_var_add_backward);
}

rac_var_t *rac_var_mul(rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return rac_var_make_ex(lhs->alloctr, lhs->data * rhs->data, '*', (rac_var_t*[2]){lhs, rhs}, rac_var_mul_backward);
}

rac_var_t *rac_var_div(rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return rac_var_make_ex(lhs->alloctr, lhs->data / rhs->data, '/', (rac_var_t*[2]){lhs, rhs}, rac_var_mul_backward);
}

void rac_var_add_inplace(rac_var_t *out, rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(out != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    
    // remake with updated variables
    rac_var_remake(out, lhs->data + rhs->data, '+', (rac_var_t*[2]){lhs, rhs}, rac_var_add_backward);
}

void rac_var_sub_inplace(rac_var_t *out, rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(out != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // remake with updated variables
    rac_var_remake(out, lhs->data - rhs->data, '-', (rac_var_t*[2]){lhs, rhs}, rac_var_add_backward);
}

void rac_var_mul_inplace(rac_var_t *out, rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(out != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // remake with updated variables
    rac_var_remake(out, lhs->data * rhs->data, '*', (rac_var_t*[2]){lhs, rhs}, rac_var_mul_backward); 
}

void rac_var_div_inplace(rac_var_t *out, rac_var_t *const lhs, rac_var_t *const rhs) {
    // check for invalid input
    VT_DEBUG_ASSERT(out != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(lhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(rhs != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // remake with updated variables
    rac_var_remake(out, lhs->data / rhs->data, '/', (rac_var_t*[2]){lhs, rhs}, rac_var_mul_backward); 
}

void rac_var_update(rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // update
    if (var->parents[0] && var->parents[1]) {
        rac_var_t *lhs = var->parents[0];
        rac_var_t *rhs = var->parents[1];
        switch(var->op) {
            case '+': rac_var_add_inplace(var, lhs, rhs); break;
            case '-': rac_var_sub_inplace(var, lhs, rhs); break;
            case '*': rac_var_mul_inplace(var, lhs, rhs); break;
            case '/': rac_var_div_inplace(var, lhs, rhs); break;
            default: break;
        }
    }

    // zero grad
    rac_var_zero_grad(var);
}

/* 
    Other
*/

vt_plist_t *rac_var_build_parent_tree(rac_var_t *const node_start) {
    // check for invalid input
    VT_DEBUG_ASSERT(node_start != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // create node list
    vt_plist_t *node_list = vt_plist_create(VT_ARRAY_DEFAULT_INIT_ELEMENTS, node_start->alloctr);

    // walk all tree nodes
    rac_var_deep_walk(node_start, node_list);

    return node_list;
}

// -------------------------- PRIVATE -------------------------- //

/**
 * @brief Builds parent (dependency) tree
 * @param node_curr current node
 * @param node_list node list
 * @returns None
 */
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

/**
 * @brief Performs backward operation on addition and substraction
 * @param op_result addition/substraction operation result
 * @returns None
 */
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

/**
 * @brief Performs backward operation on multiplication and division
 * @param op_result multiplication/division operation result
 * @returns None
 */
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

