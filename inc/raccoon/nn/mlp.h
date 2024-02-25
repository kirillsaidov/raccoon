#ifndef RACCOON_NN_MLP_H
#define RACCOON_NN_MLP_H

/** MLP MODULE (multi-layer perceptron)
 * Functions:
    - rac_mlp_make
    - rac_mlp_make_ex
    - rac_mlp_free
    - rac_mlp_forward
    - rac_mlp_zero_grad
    - rac_mlp_update
*/

#include "raccoon/nn/layer.h"

// MLP with neurons
typedef struct RaccoonMLP {
    // layers
    vt_plist_t *layers;

    // allocator: if `NULL`, then calloc/realloc/free is used
    struct VitaBaseAllocatorType *alloctr;
} rac_mlp_t;

/* 
    MLP creation/destruction
*/

/**
 * @brief  Creates an MLP model
 * @param alloctr allocator instance
 * @param num_layers number of layers including the input layer (shape length)
 * @param shape expected input size
 * @param activate_hidden activation function for hidden layers; if linear use `NULL`
 * @param activate_output activation function for the output layer; if linear use `NULL`
 * @returns valid `rac_mlp_t*` or asserts on failure
 */
extern rac_mlp_t *rac_mlp_make(
    struct VitaBaseAllocatorType *const alloctr, 
    const size_t num_layers, 
    const size_t shape[], 
    rac_var_t *(*activate_hidden)(rac_var_t *const), 
    rac_var_t *(*activate_output)(rac_var_t *const)
);

/**
 * @brief  Creates an MLP model
 * @param alloctr allocator instance
 * @param layers list of layers initialized
 * @returns valid `rac_mlp_t*` or asserts on failure
 * 
 * @note mlp frees `layers` automatically
 */
extern rac_mlp_t *rac_mlp_make_ex(struct VitaBaseAllocatorType *const alloctr, vt_plist_t *const layers);

/**
 * @brief  Frees a mlp instance
 * @param mlp instance
 * @returns None
 */
extern void rac_mlp_free(rac_mlp_t *mlp);

/* 
    MLP operations
*/

/**
 * @brief  Forward operation
 * @param mlp instance
 * @param input ditto
 * @returns valid `vt_plist_t*` of `rac_var_t*` or asserts on failure
 */
extern vt_plist_t *rac_mlp_forward(rac_mlp_t *const mlp, const vt_plist_t *const input);

/**
 * @brief  Zero all gradients
 * @param mlp instance
 * @returns None
 */
extern void rac_mlp_zero_grad(rac_mlp_t *const mlp);

/**
 * @brief  Update mlp parameters
 * @param mlp instance
 * @param lr learning rate
 * @returns None
 */
extern void rac_mlp_update(rac_mlp_t *const mlp, const rac_float lr);

#endif // RACCOON_NN_MLP_H

