#ifndef RACCOON_NN_LAYER_H
#define RACCOON_NN_LAYER_H

/** LAYER MODULE
 * Functions:
    - rac_layer_make
    - rac_layer_free
    - rac_layer_forward
    - rac_layer_zero_grad
    - rac_layer_update
*/

#include "raccoon/nn/neuron.h"

// Layer with neurons
typedef struct RaccoonLayer {
    // neurons
    vt_plist_t *neurons;

    // model last output (predictions)
    vt_plist_t *last_prediction;

    // activation funtions (it must set the backward function to be used in backward propagation)
    rac_var_t *(*activate)(rac_var_t *const);

    // allocator: if `NULL`, then calloc/realloc/free is used
    struct VitaBaseAllocatorType *alloctr;
} rac_layer_t;

/* 
    Layer creation/destruction
*/

/**
 * @brief  Creates a layer
 * @param alloctr allocator instance
 * @param input_size expected input size
 * @param output_size expected input size
 * @param activate activation function; if linear use `NULL`
 * @returns valid `rac_layer_t*` or asserts on failure
 */
extern rac_layer_t *rac_layer_make(struct VitaBaseAllocatorType *const alloctr, const size_t input_size, const size_t output_size, rac_var_t *(*activate)(rac_var_t *const));

/**
 * @brief  Frees a layer instance
 * @param layer instance
 * @returns None
 */
extern void rac_layer_free(rac_layer_t *layer);

/* 
    Layer operations
*/

/**
 * @brief  Forward operation
 * @param layer instance
 * @param input ditto
 * @returns valid `vt_plist_t*` of `rac_var_t*` or asserts on failure
 */
extern vt_plist_t *rac_layer_forward(rac_layer_t *const layer, const vt_plist_t *const input);

/**
 * @brief  Zero all gradients
 * @param layer instance
 * @returns None
 */
extern void rac_layer_zero_grad(rac_layer_t *const layer);

/**
 * @brief  Update layer parameters
 * @param layer instance
 * @param lr learning rate
 * @returns None
 */
extern void rac_layer_update(rac_layer_t *const layer, const rac_float lr);

#endif // RACCOON_NN_LAYER_H

