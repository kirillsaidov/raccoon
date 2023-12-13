#include "raccoon/core/core.h"

// generate prisma error strings
#define X(a) VT_STRING_OF(a),
static const char *const rac_error_str[] = {
    RAC_i_GENERATE_RAC_STATUS(X)
};
#undef X

const char *rac_status_to_str(const enum RaccoonStatus e) {
    if (e < RAC_STATUS_COUNT) {
        return rac_error_str[e];
    }

    return NULL;
}

