#ifndef RACCOON_AUXILIARY_LOSS_H
#define RACCOON_AUXILIARY_LOSS_H

/** LOSS MODULE
 * Functions:
    - 
*/

#include "raccoon/core/core.h"
#include "raccoon/core/variable.h"
#include "vita/container/plist.h"

/**
 * @brief  Calculates L2 loss
 * @param  loss valid variable
 * @param  pred list of predictions
 * @param  target list of target data
 * @returns rac_float accuracy
 */
extern rac_float rac_loss_l2(rac_var_t *const loss, const vt_plist_t *const pred, const vt_plist_t *const target);

#endif // RACCOON_AUXILIARY_LOSS_H

