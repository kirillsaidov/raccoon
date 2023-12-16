#ifndef RACCOON_CORE_VARIABLE_H
#define RACCOON_CORE_VARIABLE_H

/** VARIABLE MODULE
 * Functions:
    - rac_var_make
    - rac_var_make_ex
    - rac_var_make_rand
    - rac_var_free
    - rac_var_backward
    - rac_var_zero_grad
    - rac_var_add
    - rac_var_sub
    - rac_var_mul
    - rac_var_div
    - rac_var_build_parent_tree
*/

#include "raccoon/core/core.h"
#include "vita/container/plist.h"

// parent node length (there can only be two parents at a time)
#define RAC_VAR_PARENTS_LEN 2

// Variable with autograd functionality
typedef struct RaccoonVariable {
    // numerical data
    rac_float data;
 
    // gradient value
    rac_float grad;

    // parent nodes
    struct RaccoonVariable *parents[RAC_VAR_PARENTS_LEN];

    // backward function
    void (*backward)(struct RaccoonVariable*);

    // allocator: if `NULL`, then calloc/realloc/free is used
    struct VitaBaseAllocatorType *alloctr;
} rac_var_t;

/* 
    Variable creation/destruction
*/

/**
 * @brief  Creates a variable
 * @param  alloctr allocator instance
 * @param  data numerical data
 * @returns valid `rac_var_t*` or asserts on failure
 */
extern rac_var_t *rac_var_make(struct VitaBaseAllocatorType *const alloctr, const rac_float data);

/**
 * @brief  Creates a variable, extended
 * @param  alloctr allocator instance
 * @param  data numerical data
 * @param  parents parent nodes
 * @param  backward backward function
 * @returns valid `rac_var_t*` or asserts on failure
 */
extern rac_var_t *rac_var_make_ex(struct VitaBaseAllocatorType *const alloctr, const rac_float data, struct RaccoonVariable *parents[2], void (*backward)(struct RaccoonVariable*));

/**
 * @brief  Creates a random variable in range [0; 1)
 * @param  alloctr allocator instance
 * @returns valid `rac_var_t*` or asserts on failure
 */
extern rac_var_t *rac_var_make_rand(struct VitaBaseAllocatorType *const alloctr);

/**
 * @brief  Frees a variable instance
 * @param  var variable instance
 * @returns None
 */
extern void rac_var_free(rac_var_t *var);

/* 
    Variable operations
*/

/**
 * @brief  Perform backward propagation
 * @param  var variable instance
 * @returns None
 */
extern void rac_var_backward(rac_var_t *const var);

/**
 * @brief  Zero all gradients
 * @param  var variable instance
 * @returns None
 */
extern void rac_var_zero_grad(rac_var_t *const var);

/**
 * @brief  Add two variables
 * @param  rhs variable instance
 * @returns valid `rac_var_t*` or asserts on failure
 */
extern rac_var_t *rac_var_add(rac_var_t *const lhs, rac_var_t *const rhs);

/**
 * @brief  Substract two variables
 * @param  lhs variable instance
 * @param  rhs variable instance
 * @returns valid `rac_var_t*` or asserts on failure
 */
extern rac_var_t *rac_var_sub(rac_var_t *const lhs, rac_var_t *const rhs);

/**
 * @brief  Multiply two variables
 * @param  lhs variable instance
 * @param  rhs variable instance
 * @returns valid `rac_var_t*` or asserts on failure
 */
extern rac_var_t *rac_var_mul(rac_var_t *const lhs, rac_var_t *const rhs);

/**
 * @brief  Divide two variables
 * @param  rhs variable instance
 * @param  lhs variable instance
 * @returns valid `rac_var_t*` or asserts on failure
 */
extern rac_var_t *rac_var_div(rac_var_t *const lhs, rac_var_t *const rhs);

/* 
    Other
*/

/**
 * @brief  Builds parent (dependency) tree
 * @param  node_start start from node
 * @returns a list of parents including the starting node or asserts on failure
 */
extern vt_plist_t *rac_var_build_parent_tree(rac_var_t *const node_start);

#endif // RACCOON_CORE_VARIABLE_H

