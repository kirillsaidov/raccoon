#include "raccoon/raccoon.h"
#include "vita/vita.h"

// test suite
static int test_num = 0;
#define TEST(func) { printf("(%d) ---> TESTING: %s\n", test_num, #func); func(); test_num++; }

/**
 * TESTS
 */

void test_var(void);
void test_neuron(void);
void test_layer(void);
void test_mlp(void);

/**
 * HELPER FUNCTIONS
 */

void plist_var_free(vt_plist_t *list);

static vt_mallocator_t *alloctr = NULL;
int main(void) {
    vt_version_t 
        vt_v = vt_version_get(),
        rac_v = rac_version_get();
    printf("Vita (%s) | Raccoon (%s)\n", vt_v.str, rac_v.str);

    // start
    alloctr = vt_mallocator_create();
    {
        vt_debug_disable_output(true);
        // TEST(test_var);
        // TEST(test_neuron);
        // TEST(test_layer);
        TEST(test_mlp);
    }
    vt_mallocator_print_stats(alloctr->stats);
    vt_mallocator_destroy(alloctr);

    return 0;
}

/**
 * TESTS
 */

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
    assert(c->backward != NULL);

    // backward
    rac_var_backward(c);
    assert(c->grad == 1);
    assert(a->grad == 1);
    assert(b->grad == 1);

    // zero grad
    rac_var_zero_grad(c);
    rac_var_zero_grad(b);
    rac_var_zero_grad(a);
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
    assert(c->backward != NULL);

    // backward
    rac_var_backward(c);
    assert(c->grad == 1);
    assert(a->grad == 1);
    assert(b->grad == 1);

    // zero grad
    rac_var_zero_grad(c);
    rac_var_zero_grad(b);
    rac_var_zero_grad(a);
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
    assert(c->backward != NULL);

    // backward
    rac_var_backward(c);
    assert(c->grad == 1);
    assert(a->grad == 3);
    assert(b->grad == 2);

    // zero grad
    rac_var_zero_grad(c);
    rac_var_zero_grad(b);
    rac_var_zero_grad(a);
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
    assert(c->backward != NULL);

    // backward
    rac_var_backward(c);
    assert(c->grad == 1);
    assert(a->grad == 3);
    assert(b->grad == 6);

    // zero grad
    rac_var_zero_grad(c);
    rac_var_zero_grad(b);
    rac_var_zero_grad(a);
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
    rac_var_zero_grad(a);
    rac_var_zero_grad(b);
    rac_var_zero_grad(c);
    rac_var_zero_grad(d);
    rac_var_zero_grad(e);
    rac_var_zero_grad(f);
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

void test_neuron(void) {
    // allocate, test, free
    const size_t input_size = 2;
    rac_neuron_t *perceptron = rac_neuron_make(alloctr, input_size, NULL);
    assert(perceptron->params != NULL);
    assert(perceptron->cache != NULL);
    assert(perceptron->activate == NULL);
    assert(vt_plist_len(perceptron->params) == input_size+1);
    assert(vt_plist_capacity(perceptron->cache) == 2*(input_size+1)+2);
    rac_neuron_free(perceptron);

    /**
     * FORWARD PROPAGATION
     */

    // input
    rac_var_t *target = rac_var_make(alloctr, 4);
    vt_plist_t *input = vt_plist_create(input_size, alloctr);
    vt_plist_push_back(input, rac_var_make(alloctr, 1)); // x1
    vt_plist_push_back(input, rac_var_make(alloctr, 2)); // x2

    // params
    vt_plist_t *params = vt_plist_create(input_size+1, alloctr);
    vt_plist_push_back(params, rac_var_make(alloctr, 0.5)); // w1
    vt_plist_push_back(params, rac_var_make(alloctr, 0.5)); // w2
    vt_plist_push_back(params, rac_var_make(alloctr, 0.5)); // b

    // y = w1 * x1 + w2 * x2 + b
    perceptron = rac_neuron_make_ex(alloctr, params, NULL);
    assert(perceptron->params != NULL);
    assert(perceptron->cache != NULL);
    assert(perceptron->activate == NULL);
    assert(vt_plist_len(perceptron->params) == input_size+1);
    assert(vt_plist_capacity(perceptron->cache) == 2*(input_size+1)+2);

    // forward
    rac_var_t *pred = rac_neuron_forward(perceptron, input);
    assert(pred->data == 2);

    // loss
    rac_var_t *loss = rac_var_sub(target, pred);
    assert(loss->data == 2);

    // backward
    rac_var_backward(loss);
    assert(((rac_var_t*)vt_plist_get(params, 0))->grad == ((rac_var_t*)vt_plist_get(input, 0))->data);
    assert(((rac_var_t*)vt_plist_get(params, 1))->grad == ((rac_var_t*)vt_plist_get(input, 1))->data);
    assert(((rac_var_t*)vt_plist_get(params, 2))->grad == 1);

    // loop
    const size_t iters = 100;
    VT_FOREACH(epoch, 0, iters) {
        // forward
        rac_var_t *yhat = rac_neuron_forward(perceptron, input);

        // loss
        rac_var_t *cost = rac_var_sub(yhat, target);

        // backward
        rac_neuron_zero_grad(perceptron);
        rac_var_backward(cost);

        // update
        const rac_float lr = 0.05;
        const rac_float cost_data = cost->data;
        rac_neuron_update(perceptron, lr * cost->data);

        // free cost
        rac_var_free(cost);

        // stop
        if (cost_data > -0.01 && cost_data < 0.01) break;

        // print progress
        if (epoch % 1 == 0) {
            printf("step %zu loss %.2f lr %.2f\n", epoch, cost_data, lr);
            printf("w0: %.2f | grad: %.2f\n", ((rac_var_t*)vt_plist_get(params, 0))->data, ((rac_var_t*)vt_plist_get(params, 0))->grad);
            printf("w1: %.2f | grad: %.2f\n", ((rac_var_t*)vt_plist_get(params, 1))->data, ((rac_var_t*)vt_plist_get(params, 1))->grad);
            printf("b : %.2f | grad: %.2f\n\n", ((rac_var_t*)vt_plist_get(params, 2))->data, ((rac_var_t*)vt_plist_get(params, 2))->grad);
        }
    }

    // test model
    pred = rac_neuron_forward(perceptron, input);
    assert(vt_math_is_close(pred->data, 4, 0.01));

    // zero grad
    rac_neuron_zero_grad(perceptron);
    assert(((rac_var_t*)vt_plist_get(params, 0))->grad == 0);
    assert(((rac_var_t*)vt_plist_get(params, 1))->grad == 0);

    // free only the things you've allocated yourself
    rac_var_free(loss);
    rac_var_free(target);
    plist_var_free(input);
    rac_neuron_free(perceptron);
    // plist_var_free(params);  // no need to free, is freed by the neuron, since it was passed in as a parameter
    // rac_var_free(pred);      // no need to free, is freed by the neuron, since it was allocated by it
}

void test_layer(void) {
    // allocate, test, free
    rac_layer_t *layer = rac_layer_make(alloctr, 4, 1, NULL);
    assert(layer->neurons != NULL);
    assert(layer->last_prediction != NULL);
    assert(layer->activate == NULL);
    assert(vt_plist_len(layer->neurons) == 1);
    assert(vt_plist_capacity(layer->last_prediction) == 1);
    rac_layer_free(layer);

    /**
     * FORWARD PROPAGATION
     */

    /* --- INIT --- */

    // input
    const size_t input_size = 3;
    const size_t input_rows = 8;
    vt_plist_t *cache = vt_plist_create(input_rows, alloctr);
    vt_plist_t *input = vt_plist_create(input_rows, alloctr);
    vt_plist_t *target = vt_plist_create(input_rows, alloctr);
    
    // define data
    const rac_float data[input_rows * (input_size+1)] = {
        // x     y
        0, 0, 0, 1,
        0, 0, 1, 0,
        0, 1, 1, 0,
        1, 1, 1, 0,
        1, 1, 0, 1,
        1, 0, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1,
    };
    VT_FOREACH(i, 0, input_rows) {
        vt_plist_t *x = vt_plist_create(input_size, alloctr);
        VT_FOREACH(j, 0, input_size) vt_plist_push_back(x, rac_var_make(alloctr, data[vt_index_2d_to_1d(i, j, 4)]));
        vt_plist_push_back(input, x);
        vt_plist_push_back(target, rac_var_make(alloctr, data[vt_index_2d_to_1d(i, 3, 4)]));
    }
    
    // model
    layer = rac_layer_make(alloctr, input_size, 1, NULL);

    // size
    rac_var_t *batch_size = rac_var_make(alloctr, input_rows);

    /* --- FORWARD --- */

    // loop
    const rac_float lr = 0.005;
    const size_t iters = 100;
    VT_FOREACH(epoch, 0, iters) {
        // batch forward
        rac_float accuracy = 0;
        rac_var_t *loss = rac_var_make(alloctr, 0); vt_plist_push_front(cache, loss);
        VT_FOREACH(i, 0, input_rows) {
            // forward
            vt_plist_t *x = vt_plist_get(input, i);
            vt_plist_t *out = rac_layer_forward(layer, x);
            rac_var_t *yhat = vt_plist_get(out, 0);
            rac_var_t *ytarget = vt_plist_get(target, i);

            // loss: mse
            loss = rac_var_sub(yhat, ytarget); vt_plist_push_back(cache, loss);
            loss = rac_var_mul(loss, loss); vt_plist_push_back(cache, loss);
            loss = rac_var_add(loss, vt_plist_get(cache, 0)); vt_plist_push_front(cache, loss);
            
            // accuracy
            accuracy += ((yhat->data > 0.5) == ytarget->data);
        }
        loss = rac_var_div(loss, batch_size); vt_plist_push_back(cache, loss);
        accuracy /= input_rows;

        // backward
        rac_layer_zero_grad(layer);
        rac_var_backward(loss);

        // update
        rac_layer_update(layer, lr);

        // output progress
        if (epoch % 10 == 0) printf("epoch %3zu loss %.4f accuracy %.4f\n", epoch, loss->data, accuracy);
    }

    // free
    plist_var_free(cache);
    plist_var_free(target);
    VT_FOREACH(i, 0, vt_plist_len(input)) plist_var_free(vt_plist_get(input, i));
    vt_plist_destroy(input);
    rac_layer_free(layer);
    rac_var_free(batch_size);
}

void test_mlp(void) {
    // allocate, test, free
    rac_mlp_t *model = rac_mlp_make(alloctr, 3, (size_t[]) {2, 4, 1}, NULL, NULL);
    assert(model->layers != NULL);
    assert(vt_plist_len(model->layers) == 2);
    rac_mlp_free(model);

    /**
     * TEST MLP MODEL
     */

    /* --- INIT --- */

    // input
    const size_t input_size = 3;
    const size_t input_rows = 8;
    vt_plist_t *cache = vt_plist_create(input_rows, alloctr);
    vt_plist_t *input = vt_plist_create(input_rows, alloctr);
    vt_plist_t *target = vt_plist_create(input_rows, alloctr);
    
    // define data
    const rac_float data[input_rows * (input_size+1)] = {
        // x     y
        0, 0, 0, 1,
        0, 0, 1, 0,
        0, 1, 1, 0,
        1, 1, 1, 0,
        1, 1, 0, 1,
        1, 0, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1,
    };
    VT_FOREACH(i, 0, input_rows) {
        vt_plist_t *x = vt_plist_create(input_size, alloctr);
        VT_FOREACH(j, 0, input_size) vt_plist_push_back(x, rac_var_make(alloctr, data[vt_index_2d_to_1d(i, j, 4)]));
        vt_plist_push_back(input, x);
        vt_plist_push_back(target, rac_var_make(alloctr, data[vt_index_2d_to_1d(i, 3, 4)]));
    }
    
    // model
    model = rac_mlp_make(alloctr, 3, (size_t[]){3, 5, 1}, NULL, NULL);

    // size
    rac_var_t *batch_size = rac_var_make(alloctr, input_rows);

    /* --- FORWARD --- */
    // loop
    const rac_float lr = 0.0005;
    const size_t iters = 100;
    VT_FOREACH(epoch, 0, iters) {
        // batch forward
        rac_float accuracy = 0;
        rac_var_t *loss = rac_var_make(alloctr, 0); vt_plist_push_front(cache, loss);
        VT_FOREACH(i, 0, input_rows) {
            // forward
            vt_plist_t *x = vt_plist_get(input, i);
            vt_plist_t *out = rac_mlp_forward(model, x);
            rac_var_t *yhat = vt_plist_get(out, 0);
            rac_var_t *ytarget = vt_plist_get(target, i);

            // loss: mse
            loss = rac_var_sub(yhat, ytarget); vt_plist_push_back(cache, loss);
            loss = rac_var_mul(loss, loss); vt_plist_push_back(cache, loss);
            loss = rac_var_add(loss, vt_plist_get(cache, 0)); vt_plist_push_front(cache, loss);
            
            // accuracy
            accuracy += ((yhat->data > 0.5) == ytarget->data);
        }
        loss = rac_var_div(loss, batch_size); vt_plist_push_back(cache, loss);
        accuracy /= input_rows;

        // backward
        rac_mlp_zero_grad(model);
        rac_var_backward(loss);

        // update
        rac_mlp_update(model, lr);

        // output progress
        if (epoch % 10 == 0) printf("epoch %3zu loss %.4f accuracy %.4f\n", epoch, loss->data, accuracy);
    }

    // free
    plist_var_free(cache);
    plist_var_free(target);
    VT_FOREACH(i, 0, vt_plist_len(input)) plist_var_free(vt_plist_get(input, i));
    vt_plist_destroy(input);
    rac_mlp_free(model);
    rac_var_free(batch_size);
}

/**
 * HELPER FUNCTIONS
 */

// frees plist and its contents
void plist_var_free(vt_plist_t *list) {
    assert(list != NULL);

    // free list contents
    VT_FOREACH(i, 0, vt_plist_len(list)) rac_var_free(vt_plist_get(list, i));

    // free list itself
    vt_plist_destroy(list);
}

