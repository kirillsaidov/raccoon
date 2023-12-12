#ifndef RACCOON_VERSION_H
#define RACCOON_VERSION_H

/** VERSION MODULE
 * This module describes library version.
 * Functions:
    - rac_version_get
*/

#include "vita/core/version.h"

// defines
#define RAC_RACCOON_VERSION_MAJOR 0
#define RAC_RACCOON_VERSION_MINOR 0
#define RAC_RACCOON_VERSION_PATCH 1
#define RAC_RACCOON_VERSION VT_STRING_OF(VT_PCAT(VT_PCAT(VT_PCAT(VT_PCAT(RAC_RACCOON_VERSION_MAJOR, .), RAC_RACCOON_VERSION_MINOR), .), RAC_RACCOON_VERSION_PATCH))

/** Query Prisma version
    @returns vt_version_t struct containing major, minor, patch and full version str
*/
extern vt_version_t rac_version_get(void);

#endif // RACCOON_VERSION_H

