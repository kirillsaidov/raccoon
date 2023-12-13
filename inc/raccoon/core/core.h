#ifndef RACCOON_CORE_H
#define RACCOON_CORE_H

/** CORE MODULE
 * Functions:
    - rac_status_to_str
*/

#include "vita/core/core.h"
#include "vita/util/debug.h"
#include "vita/allocator/mallocator.h"

#if defined(RACCOON_USE_TYPE_DOUBLE)
    #define RAC_FLOAT double
    #define RAC_ABS fabs
    #define RAC_CEIL ceil
    #define RAC_FLOOR floor
    #define RAC_ROUND round
    #define RAC_POW pow
    #define RAC_SQRT sqrt
    #define RAC_CLAMP vt_cmp_clampd
    #define RAC_MAX vt_cmp_maxd
    #define RAC_MIN vt_cmp_mind
    #define RAC_EXP exp
    #define RAC_TANH tanh
    #define RAC_LOG log
    #define RAC_CONST_EPSILON __DBL_EPSILON__
#elif defined(RACCOON_USE_TYPE_LONG_DOUBLE)
    #define RAC_FLOAT long double
    #define RAC_ABS fabsl
    #define RAC_CEIL ceill
    #define RAC_FLOOR floorl
    #define RAC_ROUND roundl
    #define RAC_POW powl
    #define RAC_SQRT sqrtl
    #define RAC_CLAMP vt_cmp_clampr
    #define RAC_MAX vt_cmp_maxr
    #define RAC_MIN vt_cmp_minr
    #define RAC_EXP expl
    #define RAC_TANH tanhl
    #define RAC_LOG logl
    #define RAC_CONST_EPSILON __LDBL_EPSILON__
#else
    #define RAC_FLOAT float
    #define RAC_ABS fabsf
    #define RAC_CEIL ceilf
    #define RAC_FLOOR floorf
    #define RAC_ROUND roundf
    #define RAC_POW powf
    #define RAC_SQRT sqrtf
    #define RAC_CLAMP vt_cmp_clampf
    #define RAC_MAX vt_cmp_maxf
    #define RAC_MIN vt_cmp_minf
    #define RAC_EXP expf
    #define RAC_TANH tanhf
    #define RAC_LOG logf
    #define RAC_CONST_EPSILON __FLT_EPSILON__
#endif
typedef RAC_FLOAT rac_float;

// prisma error codes
#define RAC_i_GENERATE_RAC_STATUS(apply) \
    apply(RAC_STATUS_ERROR_IS_NULL)                  /* element wasn't initialized or is NULL */ \
    apply(RAC_STATUS_ERROR_IS_REQUIRED)              /* precondition is required */ \
    apply(RAC_STATUS_ERROR_ALLOCATION)               /* failed to allocate or reallocate memory */ \
    apply(RAC_STATUS_ERROR_INVALID_ARGUMENTS)        /* invalid arguments supplied */ \
    apply(RAC_STATUS_ERROR_OUT_OF_BOUNDS_ACCESS)     /* accessing memory beyond allocated size */ \
    apply(RAC_STATUS_ERROR_INCOMPATIBLE_SHAPES)      /* incompatible tensor shape */ \
    apply(RAC_STATUS_ERROR_INCOMPATIBLE_DIMENSIONS)  /* different dimensions */ \
    apply(RAC_STATUS_OPERATION_FAILURE)              /* failed to perform an action */ \
    apply(RAC_STATUS_OPERATION_SUCCESS)              /* all good */ \
    apply(RAC_STATUS_COUNT)                          /* number of elements */

// generate prisma error codes
#define X(a) a,
enum RaccoonStatus {
    RAC_i_GENERATE_RAC_STATUS(X)
};
#undef X

/**
 * @brief  Returns a Raccoon error string from prisma error code
 * @param  e raccoon error code
 * @returns C string upon success, `NULL` otherwise
 */
extern const char *rac_status_to_str(const enum RaccoonStatus e);

#endif // RACCOON_CORE_H

