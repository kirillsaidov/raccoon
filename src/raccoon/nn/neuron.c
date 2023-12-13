#include "raccoon/nn/neuron.h"
#include "vita/math/math.h"

/* 
    Neuron creation/destruction
*/

rac_neuron_t *rac_neuron_make(struct VitaBaseAllocatorType *const alloctr, const size_t input_size, rac_var_t *(*activate)(rac_var_t *const)) {
    // allocate neuron instance
    rac_neuron_t *neuron = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_neuron_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_neuron_t));

    // init neuron
    *neuron = (rac_neuron_t) {
        .params = vt_plist_create(input_size, alloctr),
        .activate = activate,
        .alloctr = alloctr,
    };
    
    // init weights + bias
    VT_FOREACH(i, 0, input_size+1) {
        vt_plist_push_back(neuron->params, rac_var_make(alloctr, vt_math_random_f32_uniform(0, 1)));
    }

    return neuron;
}

rac_neuron_t *rac_neuron_make_ex(struct VitaBaseAllocatorType *const alloctr, vt_plist_t *const params, rac_var_t *(*activate)(rac_var_t *const)) {
    // check for invalid input
    VT_DEBUG_ASSERT(params != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(vt_plist_len(params) >= 2, "%s: At least 1 weight and 1 bias is required!\n", rac_status_to_str(RAC_STATUS_ERROR_IS_REQUIRED));

    // allocate neuron instance
    rac_neuron_t *neuron = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_neuron_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_neuron_t));

    // init neuron
    *neuron = (rac_neuron_t) {
        .params = params,
        .activate = activate,
        .alloctr = alloctr,
    };

    return neuron;
}

void rac_neuron_free(rac_neuron_t *neuron) {
    // check for invalid input
    VT_DEBUG_ASSERT(neuron != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // free all nodes params
    const size_t params_len = vt_plist_len(neuron->params);
    VT_FOREACH(i, 0, params_len) rac_var_free(vt_plist_get(neuron->params, i));
    vt_plist_destroy(neuron->params);

    // free neuron
    (neuron->alloctr) ? VT_ALLOCATOR_FREE(neuron->alloctr, neuron) : VT_FREE(neuron);

    // reset
    *((rac_neuron_t**)neuron) = NULL;
}

/* 
    Neuron operations
*/

rac_var_t *rac_neuron_forward(rac_neuron_t *const neuron, const vt_plist_t *const input) {
    // check for invalid input
    VT_DEBUG_ASSERT(neuron != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(input != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_ENFORCE(vt_plist_len(input) == vt_plist_len(neuron->params)-1, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // forward
    rac_var_t *sum = rac_var_make(NULL, 0);
    const size_t input_size = vt_plist_len(input);
    VT_FOREACH(i, 0, input_size) {
        rac_var_t *tmp = rac_var_mul(vt_plist_get(neuron->params, i), vt_plist_get(input, i));
        sum = rac_var_add(sum, tmp);
    }
    sum = rac_var_add(sum, vt_plist_get(neuron->params, input_size)); // add bias

    // activate
    rac_var_t *result = NULL;
    if (neuron->activate) result = neuron->activate(sum);

    return result;
}

void rac_neuron_zero_grad(rac_neuron_t *const neuron) {
    // check for invalid input
    VT_DEBUG_ASSERT(neuron != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // zero out the gradients (weights + bias)
    const size_t params_len = vt_plist_len(neuron->params);
    VT_FOREACH(i, 0, params_len) rac_var_zero_grad(vt_plist_get(neuron->params, i));
}

