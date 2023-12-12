#ifndef RACCOON_OP_BACKWARD_H
#define RACCOON_OP_BACKWARD_H

/** OP_BACKWARD MODULE (basic operations)
 * This module contains backward functions on basic operations
 * Functions:
    - //
*/

#include "raccoon/core.h"

// checkout raccoon/variable.h
typedef struct RaccoonVariable rac_var_t;

/**
 * @brief  Backward operation on addition
 * @param  op_result the resulting `var_rac_t` instance from addition operation
 * @returns None
 */
extern void rac_op_backward_add(rac_var_t *const op_result);

/**
 * @brief  Backward operation on multiplication
 * @param  op_result the resulting `var_rac_t` instance from multiplication operation
 * @returns None
 */
extern void rac_op_backward_mul(rac_var_t *const op_result);

#endif // RACCOON_OP_BACKWARD_H

