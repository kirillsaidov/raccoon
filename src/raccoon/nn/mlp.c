#include "raccoon/nn/mlp.h"

/* 
    MLP creation/destruction
*/

rac_mlp_t *rac_mlp_make(
    struct VitaBaseAllocatorType *const alloctr, 
    const size_t num_layers, 
    const size_t shape[], 
    rac_var_t *(*activate_hidden)(rac_var_t *const), 
    rac_var_t *(*activate_output)(rac_var_t *const)
) {
    // check for invalid input
    VT_DEBUG_ASSERT(num_layers > 0, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(shape != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // allocate mlp instance
    rac_mlp_t *mlp = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_mlp_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_mlp_t));

    // init mlp
    *mlp = (rac_mlp_t) {
        .layers = vt_plist_create(num_layers, alloctr),
        .alloctr = alloctr,
    };

    // init layers
    VT_FOREACH(i, 1, num_layers) {
        vt_plist_push_back(
            mlp->layers, 
            rac_layer_make(alloctr, shape[i-1], shape[i], i+1 != num_layers ? activate_hidden : activate_output)
        );
    }

    return mlp;
}

rac_mlp_t *rac_mlp_make_ex(struct VitaBaseAllocatorType *const alloctr, vt_plist_t *const layers) {
    // check for invalid input
    VT_DEBUG_ASSERT(layers != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // allocate mlp instance
    rac_mlp_t *mlp = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_mlp_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_mlp_t));

    // init mlp
    *mlp = (rac_mlp_t) {
        .layers = layers,
        .alloctr = alloctr,
    };

    return mlp;
}

void rac_mlp_free(rac_mlp_t *mlp) {
    // check for invalid input
    VT_DEBUG_ASSERT(mlp != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // free all layers
    const size_t layers_len = vt_plist_len(mlp->layers);
    VT_FOREACH(i, 0, layers_len) rac_layer_free(vt_plist_get(mlp->layers, i));
    vt_plist_destroy(mlp->layers);

    // free mlp
    (mlp->alloctr) ? VT_ALLOCATOR_FREE(mlp->alloctr, mlp) : VT_FREE(mlp);
}

/* 
    MLP operations
*/

vt_plist_t *rac_mlp_forward(rac_mlp_t *const mlp, const vt_plist_t *const input) {
    // check for invalid input
    VT_DEBUG_ASSERT(mlp != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(input != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // check shape
    const size_t input_size = vt_plist_len(input);
    rac_layer_t *layer = vt_plist_get(mlp->layers, 0);
    rac_neuron_t *neuron = vt_plist_get(layer->neurons, 0);
    const size_t layer_input_size = vt_plist_len(neuron->params);
    VT_ENFORCE(input_size+1 == layer_input_size, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // forward
    vt_plist_t *output = (vt_plist_t*)input;
    const size_t layers_len = vt_plist_len(mlp->layers);
    VT_FOREACH(i, 0, layers_len) {
        rac_layer_t *l = vt_plist_get(mlp->layers, i);
        output = rac_layer_forward(l, output);
    }

    return output;
}

void rac_mlp_zero_grad(rac_mlp_t *const mlp) {
    // check for invalid input
    VT_DEBUG_ASSERT(mlp != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // zero out all gradients
    const size_t layers_len = vt_plist_len(mlp->layers);
    VT_FOREACH(i, 0, layers_len) rac_layer_zero_grad(vt_plist_get(mlp->layers, i));
}

void rac_mlp_update(rac_mlp_t *const mlp, const rac_float lr) {
    // check for invalid input
    VT_DEBUG_ASSERT(mlp != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // zero out all gradients
    const size_t layers_len = vt_plist_len(mlp->layers);
    VT_FOREACH(i, 0, layers_len) rac_layer_update(vt_plist_get(mlp->layers, i), lr);
}

