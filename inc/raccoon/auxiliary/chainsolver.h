#ifndef RACCOON_AUXILIARY_CHAINSOLVER_H
#define RACCOON_AUXILIARY_CHAINSOLVER_H

/** CHAINSOLVER MODULE
 * Functions:
    - rac_chainsolver_make 
    - rac_chainsolver_make_ex 
    - rac_chainsolver_free 
    - rac_chainsolver_add 
    - rac_chainsolver_sub
    - rac_chainsolver_mul 
    - rac_chainsolver_div 
    - rac_chainsolver_add_v 
    - rac_chainsolver_sub_v
    - rac_chainsolver_mul_v
    - rac_chainsolver_div_v
    - rac_chainsolver_push
    - rac_chainsolver_push_v
    - rac_chainsolver_reset
    - rac_chainsolver_result
*/

#include "raccoon/core/core.h"
#include "raccoon/core/variable.h"
#include "vita/container/plist.h"

// ChainSolver
typedef struct ChainSolver {
    // current solver size
    size_t chain_size;

    // cache for `rac_var_t*`
    vt_plist_t *list;

    // allocator: if `NULL`, then calloc/realloc/free is used
    struct VitaBaseAllocatorType *alloctr;
} rac_chainsolver_t;

/* 
    ChainSolver creation/destruction
*/

/**
 * @brief  Creates an empty ChainSolver instance
 * @param  alloctr allocator instance
 * @returns valid `rac_chainsolver_t*` instance
 */
extern rac_chainsolver_t *rac_chainsolver_make(struct VitaBaseAllocatorType *const alloctr);

/**
 * @brief  Creates an empty ChainSolver instance with custom initial value
 * @param  alloctr allocator instance
 * @param  init initial value
 * @returns valid `rac_chainsolver_t*` instance
 */
extern rac_chainsolver_t *rac_chainsolver_make_ex(struct VitaBaseAllocatorType *const alloctr, const rac_float init);

/**
 * @brief  Free ChainSolver instance
 * @param  solver ChainSolver instance
 * @returns None
 */
extern void rac_chainsolver_free(rac_chainsolver_t *solver);

/* 
    ChainSolver operations
*/

/**
 * @brief  Add variable to the last ChainSolver result
 * @param  solver ChainSolver instance 
 * @param  var variable instance
 * @returns None
 */
extern void rac_chainsolver_add(rac_chainsolver_t *const solver, rac_var_t *const var);

/**
 * @brief  Substract variable from the last ChainSolver result
 * @param  solver ChainSolver instance 
 * @param  var variable instance
 * @returns None
 */
extern void rac_chainsolver_sub(rac_chainsolver_t *const solver, rac_var_t *const var);

/**
 * @brief  Multiply variable with the last ChainSolver result
 * @param  solver ChainSolver instance 
 * @param  var variable instance
 * @returns None
 */
extern void rac_chainsolver_mul(rac_chainsolver_t *const solver, rac_var_t *const var);

/**
 * @brief  Divide the last ChainSolver result with variable
 * @param  solver ChainSolver instance 
 * @param  var variable instance
 * @returns None
 */
extern void rac_chainsolver_div(rac_chainsolver_t *const solver, rac_var_t *const var);

/**
 * @brief  Add value to the last ChainSolver result
 * @param  solver ChainSolver instance 
 * @param  data value
 * @returns None
 */
extern void rac_chainsolver_add_v(rac_chainsolver_t *const solver, const rac_float data);

/**
 * @brief  Substract value from the last ChainSolver result
 * @param  solver ChainSolver instance 
 * @param  data value
 * @returns None
 */
extern void rac_chainsolver_sub_v(rac_chainsolver_t *const solver, const rac_float data);

/**
 * @brief  Multiply value with the last ChainSolver result
 * @param  solver ChainSolver instance 
 * @param  data value
 * @returns None
 */
extern void rac_chainsolver_mul_v(rac_chainsolver_t *const solver, const rac_float data);

/**
 * @brief  Divide the last ChainSolver result with value
 * @param  solver ChainSolver instance 
 * @param  data value
 * @returns None
 */
extern void rac_chainsolver_div_v(rac_chainsolver_t *const solver, const rac_float data);

/**
 * @brief  Push variable to the back of ChainSolver
 * @param  solver ChainSolver instance 
 * @param  var variable instance
 * @returns None
 */
extern void rac_chainsolver_push(rac_chainsolver_t *const solver, rac_var_t *const var);

/**
 * @brief  Push value to the back of ChainSolver
 * @param  solver ChainSolver instance 
 * @param  data variable instance
 * @returns None
 */
extern void rac_chainsolver_push_v(rac_chainsolver_t *const solver, const rac_float data);

/* 
    Other
*/

/**
 * @brief  Reset ChainSolver operations (so we can reuse the allocated variables)
 * @param  solver ChainSolver instance 
 * @param  var variable instance
 * @returns None
 */
extern void rac_chainsolver_reset(rac_chainsolver_t *const solver);

/**
 * @brief  Returns the last ChainSolver result
 * @param  solver ChainSolver instance 
 * @param  var variable instance
 * @returns valid `rac_var_t*` instance
 */
extern rac_var_t *rac_chainsolver_result(const rac_chainsolver_t *const solver);

#endif // RACCOON_AUXILIARY_CHAINSOLVER_H

