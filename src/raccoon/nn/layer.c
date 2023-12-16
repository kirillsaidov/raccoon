#include "raccoon/nn/layer.h"

/* 
    Layer creation/destruction
*/

rac_layer_t *rac_layer_make(struct VitaBaseAllocatorType *const alloctr, const size_t input_size, const size_t output_size, rac_var_t *(*activate)(rac_var_t *const)) {
    // allocate layer instance
    rac_layer_t *layer = (alloctr == NULL)
        ? VT_CALLOC(sizeof(rac_layer_t))
        : VT_ALLOCATOR_ALLOC(alloctr, sizeof(rac_layer_t));

    // init layer
    *layer = (rac_layer_t) {
        .neurons = vt_plist_create(output_size, alloctr),
        .cache = vt_plist_create(output_size, alloctr),
        .activate = activate,
        .alloctr = alloctr,
    };

    // init neurons
    VT_FOREACH(i, 0, output_size) {
        vt_plist_push_back(layer->neurons, rac_neuron_make(alloctr, input_size, activate));
    }

    return layer;
}

void rac_layer_free(rac_layer_t *layer) {
    // check for invalid input
    VT_DEBUG_ASSERT(layer != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // free all neurons
    const size_t neurons_len = vt_plist_len(layer->neurons);
    VT_FOREACH(i, 0, neurons_len) rac_neuron_free(vt_plist_get(layer->neurons, i));
    vt_plist_destroy(layer->neurons);

    // destroy the cache container (its contents are freed by neurons)
    vt_plist_destroy(layer->cache);

    // free layer
    (layer->alloctr) ? VT_ALLOCATOR_FREE(layer->alloctr, layer) : VT_FREE(layer);
}

/* 
    Layer operations
*/

vt_plist_t *rac_layer_forward(rac_layer_t *const layer, const vt_plist_t *const input) {
    // check for invalid input
    VT_DEBUG_ASSERT(layer != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));
    VT_DEBUG_ASSERT(input != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // check shape
    const size_t input_size = vt_plist_len(input);
    const size_t layer_input_size = vt_plist_len(((rac_neuron_t*)vt_plist_get(layer->neurons, 0))->params);
    VT_ENFORCE(input_size+1 == layer_input_size, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INCOMPATIBLE_SHAPES));

    // clear cache
    vt_plist_clear(layer->cache);

    // forward
    const size_t neurons_len = vt_plist_len(layer->neurons);
    VT_FOREACH(i, 0, neurons_len) {
        rac_neuron_t *n = vt_plist_get(layer->neurons, i);
        vt_plist_push_back(layer->cache, rac_neuron_forward(n, input));
    }

    return layer->cache;
}

void rac_layer_zero_grad(rac_layer_t *const layer) {
    // check for invalid input
    VT_DEBUG_ASSERT(layer != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // zero out all gradients
    const size_t neurons_len = vt_plist_len(layer->neurons);
    VT_FOREACH(i, 0, neurons_len) rac_neuron_zero_grad(vt_plist_get(layer->neurons, i));
}

void rac_layer_update(rac_layer_t *const layer, const rac_float lr) {
    // check for invalid input
    VT_DEBUG_ASSERT(layer != NULL, "%s\n", rac_status_to_str(RAC_STATUS_ERROR_INVALID_ARGUMENTS));

    // zero out all gradients
    const size_t neurons_len = vt_plist_len(layer->neurons);
    VT_FOREACH(i, 0, neurons_len) rac_neuron_update(vt_plist_get(layer->neurons, i), lr);
}

