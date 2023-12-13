#ifndef RACCOON_NN_NEURON_H
#define RACCOON_NN_NEURON_H

/** NEURON MODULE
 * Functions:
    - //
*/

#include "raccoon/core/core.h"
#include "raccoon/core/variable.h"

// Neuron with weights + bias (perceptron)
typedef struct RaccoonNeuron {
    // model parameters: weights + bias
    vt_plist_t *params; 

    // activation funtions (it must set the backward function to be used in backward propagation)
    rac_var_t *(*activate)(rac_var_t *const);

    // allocator: if `NULL`, then calloc/realloc/free is used
    struct VitaBaseAllocatorType *alloctr;
} rac_neuron_t;

/* 
    Neuron creation/destruction
*/

/**
 * @brief  Creates a neuron
 * @param  alloctr allocator instance
 * @param  input_size expected input size
 * @param  activate activation function; if linear use `NULL`
 * @returns valid `rac_neuron_t*` or asserts on failure
 */
extern rac_neuron_t *rac_neuron_make(struct VitaBaseAllocatorType *const alloctr, const size_t input_size, rac_var_t *(*activate)(rac_var_t *const));

/**
 * @brief  Creates a neuron, extended
 * @param  alloctr allocator instance
 * @param  params initialized (weights + bias) parameters
 * @param  activate activation function; if linear use `NULL`
 * @returns valid `rac_neuron_t*` or asserts on failure
 */
extern rac_neuron_t *rac_neuron_make_ex(struct VitaBaseAllocatorType *const alloctr, vt_plist_t *const params, rac_var_t *(*activate)(rac_var_t *const));

/**
 * @brief  Frees a neuron instance
 * @param  neuron instance
 * @returns None
 */
extern void rac_neuron_free(rac_neuron_t *neuron);

/* 
    Neuron operations
*/

/**
 * @brief  Forward operation
 * @param  neuron instance
 * @param  input ditto
 * @returns valid `rac_var_t*` or asserts on failure
 */
extern rac_var_t *rac_neuron_forward(rac_neuron_t *const neuron, const vt_plist_t *const input);

/**
 * @brief  Zero all gradients
 * @param  neuron instance
 * @returns None
 */
extern void rac_neuron_zero_grad(rac_neuron_t *const neuron);

#endif // RACCOON_NN_NEURON_H

