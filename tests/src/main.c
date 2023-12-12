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

    /**
     * OPERATION: +
     */

    rac_var_t *a = rac_var_make(alloctr, 1);
    rac_var_t *b = rac_var_make(alloctr, 2);
    rac_var_t *c = rac_var_add(a, b);
    assert(c->data == 3);
    assert(c->grad == 0);
    assert(c->parents[0] == a && c->parents[1] == b);
    assert(c->backward == rac_op_backward_add);

    // backward
    rac_var_backward(c);
    assert(c->grad == 1);
    assert(a->grad == 1);
    assert(b->grad == 1);

    // zero grad
    rac_var_zero_grad(c);
    assert(c->grad == 0);
    assert(a->grad == 0);
    assert(b->grad == 0);

    // free
    rac_var_free(a);
    rac_var_free(b);
    rac_var_free(c);

    /**
     * OPERATION: -
     */
    
    a = rac_var_make(alloctr, 7);
    b = rac_var_make(alloctr, 2);
    c = rac_var_sub(a, b);
    assert(c->data == 5);
    assert(c->grad == 0);
    assert(c->parents[0] == a && c->parents[1] == b);
    assert(c->backward == rac_op_backward_add);

    // backward
    rac_var_backward(c);
    assert(c->grad == 1);
    assert(a->grad == 1);
    assert(b->grad == 1);

    // zero grad
    rac_var_zero_grad(c);
    assert(c->grad == 0);
    assert(a->grad == 0);
    assert(b->grad == 0);

    // free
    rac_var_free(a);
    rac_var_free(b);
    rac_var_free(c);

    /**
     * OPERATION: *
     */
    
    a = rac_var_make(alloctr, 2);
    b = rac_var_make(alloctr, 3);
    c = rac_var_mul(a, b);
    assert(c->data == 6);
    assert(c->grad == 0);
    assert(c->parents[0] == a && c->parents[1] == b);
    assert(c->backward == rac_op_backward_mul);

    // backward
    rac_var_backward(c);
    assert(c->grad == 1);
    assert(a->grad == 3);
    assert(b->grad == 2);

    // zero grad
    rac_var_zero_grad(c);
    assert(c->grad == 0);
    assert(a->grad == 0);
    assert(b->grad == 0);

    // free
    rac_var_free(a);
    rac_var_free(b);
    rac_var_free(c);

    /**
     * OPERATION: /
     */
    
    a = rac_var_make(alloctr, 6);
    b = rac_var_make(alloctr, 3);
    c = rac_var_div(a, b);
    assert(c->data == 2);
    assert(c->grad == 0);
    assert(c->parents[0] == a && c->parents[1] == b);
    assert(c->backward == rac_op_backward_mul);

    // backward
    rac_var_backward(c);
    assert(c->grad == 1);
    assert(a->grad == 3);
    assert(b->grad == 6);

    // zero grad
    rac_var_zero_grad(c);
    assert(c->grad == 0);
    assert(a->grad == 0);
    assert(b->grad == 0);

    // free
    rac_var_free(a);
    rac_var_free(b);
    rac_var_free(c);

    /**
     * TEST: a more comprehensive example
     */

    // h = ((a * b) + c) * f
    a = rac_var_make(alloctr, 2);
    b = rac_var_make(alloctr, -3);
    c = rac_var_make(alloctr, 10);
    rac_var_t *e = rac_var_mul(a, b);
    rac_var_t *d = rac_var_add(e, c);
    rac_var_t *f = rac_var_make(alloctr, -2);
    rac_var_t *g = rac_var_mul(f, d);

    // check values
    assert(g->data == -8);
    assert(f->data == -2);
    assert(d->data == 4);
    assert(e->data == -6);
    assert(c->data == 10);
    assert(b->data == -3);
    assert(a->data == 2);

    // check grad
    assert(g->grad == 0);
    assert(f->grad == 0);
    assert(d->grad == 0);
    assert(e->grad == 0);
    assert(c->grad == 0);
    assert(b->grad == 0);
    assert(a->grad == 0);

    // backward
    rac_var_backward(g);

    // check grad after backward
    assert(g->grad == 1);
    assert(f->grad == 4);
    assert(d->grad == -2);
    assert(e->grad == -2);
    assert(c->grad == -2);
    assert(b->grad == -4);
    assert(a->grad == 6);

    // zero grad
    rac_var_zero_grad(g);
    assert(g->grad == 0);
    assert(f->grad == 0);
    assert(d->grad == 0);
    assert(e->grad == 0);
    assert(c->grad == 0);
    assert(b->grad == 0);
    assert(a->grad == 0);

    // free
    rac_var_free(a);
    rac_var_free(b);
    rac_var_free(c);
    rac_var_free(d);
    rac_var_free(e);
    rac_var_free(f);
    rac_var_free(g);
}

