#include "raccoon/version.h"

vt_version_t rac_version_get(void) {
    return (vt_version_t) {
        .major = RAC_RACCOON_VERSION_MAJOR,
        .minor = RAC_RACCOON_VERSION_MINOR,
        .patch = RAC_RACCOON_VERSION_PATCH,
        .str = RAC_RACCOON_VERSION
    };
}

