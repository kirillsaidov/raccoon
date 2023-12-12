#include "raccoon/op_backward.h"
#include "raccoon/variable.h"

void rac_op_backward_add(rac_var_t *const op_result) {
    // check for invalid input
    VT_DEBUG_ASSERT(op_result != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // get lhs, rhs
    rac_var_t *const lhs = op_result->parents[0];
    rac_var_t *const rhs = op_result->parents[1];

    // perform backward operation
    lhs->grad += 1.0 * op_result->grad;
    rhs->grad += 1.0 * op_result->grad;
}

void rac_op_backward_mul(rac_var_t *const op_result) {
    // check for invalid input
    VT_DEBUG_ASSERT(op_result != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // get lhs, rhs
    rac_var_t *const lhs = op_result->parents[0];
    rac_var_t *const rhs = op_result->parents[1];

    // perform backward operation
    lhs->grad += rhs->data * op_result->grad;
    rhs->grad += lhs->data * op_result->grad;
}

