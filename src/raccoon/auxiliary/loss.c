#include "raccoon/auxiliary/loss.h"

rac_float rac_loss_l2(rac_var_t *loss, const vt_plist_t *const pred, const vt_plist_t *const target) {
    // check for invalid input
    VT_DEBUG_ASSERT(loss != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(pred != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(target != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // calculate loss and accuracy
    rac_float accuracy = 0;
    const size_t len = vt_plist_len(pred);
    VT_FOREACH(i, 0, len) {
        // retreive data
        rac_var_t *y = vt_plist_get(target, i);
        rac_var_t *yhat = vt_plist_get(pred, i);

        // calculate loss: mse
        rac_var_t *sub = rac_var_sub(yhat, y);
        rac_var_t *mul = rac_var_mul(sub, sub);
        *loss = *rac_var_add(loss, mul);

        // calculate accuracy
        accuracy += (yhat->data == y->data);
    }

    // adjust for len
    *loss = *rac_var_div(loss, rac_var_make(loss->alloctr, len));
    return accuracy/len;
}


