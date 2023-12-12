#include "raccoon/raccoon.h"

static int test_num = 0;
#define TEST(func) { printf("(%d) ---> TESTING: %s\n", test_num, #func); func(); test_num++; }

void test_var(void);

static vt_mallocator_t *alloctr = NULL;
int main(void) {
    vt_version_t 
        vt_v = vt_version_get(),
        rac_v = rac_version_get();
    printf("Vita (%s) | Raccoon (%s)\n", vt_v.str, rac_v.str);

    // start
    alloctr = vt_mallocator_create();
    {
        // vt_debug_disable_output(true);
        TEST(test_var);
    }
    vt_mallocator_print_stats(alloctr->stats);
    vt_mallocator_destroy(alloctr);

    return 0;
}

void test_var(void) {
    // allocate, test, free
    rac_var_t *var0 = rac_var_make(alloctr, 1);
    assert(var0->data == 1);
    assert(var0->grad == 0);
    assert(var0->parents[0] == NULL && var0->parents[1] == NULL);
    assert(var0->backward == NULL);
    rac_var_free(var0);

    // operations
    rac_var_t *var1 = rac_var_make(alloctr, 1);


    // zero grad
    // var1->grad = 1;
    // rac_var_zero_grad(var1);
    // assert(var1->grad == 0);
}

