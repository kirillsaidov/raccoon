#ifndef RACCOON_AUXILIARY_TAPE_H
#define RACCOON_AUXILIARY_TAPE_H

/** TAPE MODULE
 * Functions:
    - 
*/

#include "raccoon/core/core.h"
#include "raccoon/core/variable.h"
#include "vita/container/plist.h"

// Variable tape for caching operations
typedef struct RaccoonTape {
    vt_plist_t *list;
    struct VitaBaseAllocatorType *alloctr;
} rac_tape_t;

/* 
    Tape creation/destruction
*/

/**
 * @brief  Creates a tape
 * @param alloctr allocator instance
 * @returns valid `rac_tape_t*` instance
 */
extern rac_tape_t *rac_tape_make(struct VitaBaseAllocatorType *const alloctr);

/**
 * @brief  Frees tape and its elements
 * @param tape tape instance
 * @returns None
 */
extern void rac_tape_free(rac_tape_t *tape);

/* 
    Tape operations
*/

/**
 * @brief  Frees tape elements only
 * @param tape tape instance
 * @returns None
 */
extern void rac_tape_clear(rac_tape_t *const tape);

/**
 * @brief  Update tape elements values starting from the begining of the tape
 * @param tape tape instance
 * @returns None
 */
extern void rac_tape_update(rac_tape_t *const tape);

/**
 * @brief  Push an element to the tape
 * @param tape tape instance
 * @param var variable instance
 * @returns None
 */
extern void rac_tape_push(rac_tape_t *const tape, const rac_var_t *const var);

/**
 * @brief  Push multiple elements to the tape
 * @param tape tape instance
 * @param arr array of variables
 * @param arr_size number of elements to push
 * @returns None
 */
extern void rac_tape_push_ex(rac_tape_t *const tape, rac_var_t *arr[], const size_t arr_size);

/**
 * @brief  Retrive last tape value
 * @param tape tape instance
 * @returns valid `rac_var_t*` or NULL if tape length is zero
 */
extern rac_var_t *rac_tape_last(const rac_tape_t *const tape);

#endif // RACCOON_AUXILIARY_TAPE_H

