#include "raccoon/auxiliary/chainsolver.h"

/* 
    ChainSolver creation/destruction
*/

rac_chainsolver_t *rac_chainsolver_make(struct VitaBaseAllocatorType *const alloctr) {
    return rac_chainsolver_make_ex(alloctr, 0);
}

rac_chainsolver_t *rac_chainsolver_make_ex(struct VitaBaseAllocatorType *const alloctr, const rac_float init) {
    // allocate for variable
    rac_chainsolver_t *solver = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_chainsolver_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_chainsolver_t));

    // init
    *solver = (rac_chainsolver_t) {
        .list = vt_plist_create(VT_ARRAY_DEFAULT_INIT_ELEMENTS, alloctr),
        .alloctr = alloctr,
    };

    // push initial value
    rac_chainsolver_push_v(solver, init);

    return solver;
}

void rac_chainsolver_free(rac_chainsolver_t *solver) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // free solver cache list contents
    const size_t len = vt_plist_len(solver->list);
    VT_FOREACH(i, 0, len) rac_var_free(vt_plist_get(solver->list, i));

    // free solver cache list itself
    vt_plist_destroy(solver->list);

    // free solver
    (solver->alloctr) ? VT_ALLOCATOR_FREE(solver->alloctr, solver) : VT_FREE(solver);
}

/* 
    ChainSolver operations
*/

void rac_chainsolver_add(rac_chainsolver_t *const solver, rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // get last solver result
    rac_var_t *last = rac_chainsolver_result(solver);

    // push addition result to the list
    rac_chainsolver_push(solver, rac_var_add(last, var));
}

void rac_chainsolver_sub(rac_chainsolver_t *const solver, rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // get last solver result
    rac_var_t *last = rac_chainsolver_result(solver);

    // push addition result to the list
    rac_chainsolver_push(solver, rac_var_sub(last, var));
}

void rac_chainsolver_mul(rac_chainsolver_t *const solver, rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // get last solver result
    rac_var_t *last = rac_chainsolver_result(solver);

    // push addition result to the list
    rac_chainsolver_push(solver, rac_var_mul(last, var));
}

void rac_chainsolver_div(rac_chainsolver_t *const solver, rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // get last solver result
    rac_var_t *last = rac_chainsolver_result(solver);

    // push addition result to the list
    rac_chainsolver_push(solver, rac_var_div(last, var));
}

void rac_chainsolver_add_v(rac_chainsolver_t *const solver, const rac_float data) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    rac_chainsolver_add(solver, rac_var_make(solver->alloctr, data));
}

void rac_chainsolver_sub_v(rac_chainsolver_t *const solver, const rac_float data) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    rac_chainsolver_sub(solver, rac_var_make(solver->alloctr, data));
}

void rac_chainsolver_mul_v(rac_chainsolver_t *const solver, const rac_float data) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    rac_chainsolver_mul(solver, rac_var_make(solver->alloctr, data));
}

void rac_chainsolver_div_v(rac_chainsolver_t *const solver, const rac_float data) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    rac_chainsolver_div(solver, rac_var_make(solver->alloctr, data));
}

void rac_chainsolver_push(rac_chainsolver_t *const solver, rac_var_t *const var) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(var != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // push back new variable
    vt_plist_push_back(solver->list, var);    
}

void rac_chainsolver_push_v(rac_chainsolver_t *const solver, const rac_float data) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    rac_chainsolver_push(solver, rac_var_make(solver->alloctr, data));
}

/* 
    Other
*/

void rac_chainsolver_reset(rac_chainsolver_t *const solver) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // reset the first variable and free everything else
    const size_t len = vt_plist_len(solver->list);
    VT_FOREACH(i, 0, len) {
        if (i == 0) rac_var_zero_grad(vt_plist_get(solver->list, i));
        else rac_var_free(vt_plist_pop_get(solver->list));
    }
}

rac_var_t *rac_chainsolver_result(const rac_chainsolver_t *const solver) {
    // check for invalid input
    VT_DEBUG_ASSERT(solver != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    return vt_plist_get(solver->list, vt_plist_len(solver->list)-1);
}

