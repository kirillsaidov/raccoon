# Raccoon 
This is a small autograd library made for educational purposes, but works for real use cases as well! Inspired by the [Micrograd](https://github.com/karpathy/micrograd) engine.

## Features
* [Variable](inc/raccoon/core/variable.h#L25) data type with autograd
* [Neuron](inc/raccoon/nn/neuron.h#L18) perceptron model
* [Layer](inc/raccoon/nn/layer.h#L16)
* [MLP](inc/raccoon/nn/mlp.h#L17) (multi-layer perceptron)

## Getting started
```sh
# clone repo
$ git clone https://github.com/kirillsaidov/raccoon.git
$ cd raccoon

# update dependencies 
$ git submodule update --init --recursive

# build raccoon and its dependencies
$ ./build.sh  # linux, osx
$ ./build.bat # windows

# test it out (compiles and executes tests/src/main.c)
$ ./test.sh
```
Take a look at [`tests/Makefile`](tests/Makefile) to configure your build system.

## Usage example
### Variable
Below is a contrived example of possible usage:

```c
#include "raccoon/raccoon.h"

// create alloctr
vt_mallocator_t *alloctr = vt_mallocator_create();

{
    // define variables
    rac_var_t *a = rac_var_make(alloctr, 2);
    rac_var_t *b = rac_var_make(alloctr, -3);
    rac_var_t *c = rac_var_make(alloctr, 10);
    rac_var_t *f = rac_var_make(alloctr, -2);

    // h = ((a * b) + c) * f
    rac_var_t *e = rac_var_mul(a, b);
    rac_var_t *d = rac_var_add(e, c);
    rac_var_t *g = rac_var_mul(f, d);

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
    rac_var_zero_grad(g); // call for each var if needed: a, b, c, d, e, f
}

// free all data
vt_mallocator_destroy(alloctr);

// you can also free data manually
rac_var_free(a); // b, c, d, e, f, g
```

For more details check out [`tests/src/main.c`](tests/src/main.c).

### Neuron
The example below illustrates how to train a Perceptron model or Linear Regression:

```c
#include "raccoon/raccoon.h"

// model: a basic linear regression with 2 features (x1, x2), coefficients (w1, w2) and a bias (b)
// model: y = w1 * x1 + w2 * x2 + b
const size_t input_size = 2;
rac_neuron_t *perceptron = rac_neuron_make(alloctr, input_size, NULL); // activation is linear (NULL)

// define target
rac_var_t *target = rac_var_make(alloctr, 4);               // y

// define input
vt_plist_t *input = vt_plist_create(input_size, alloctr);   // create input list
vt_plist_push_back(input, rac_var_make(alloctr, 1));        // save x1 to input list
vt_plist_push_back(input, rac_var_make(alloctr, 2));        // save x2 to input list

// train model
const size_t epochs = 100;
const rac_float lr = 0.05;
rac_var_t *cost = rac_var_make(alloctr, 0);
VT_FOREACH(i, 0, epochs) {
    // forward
    rac_var_t *yhat = rac_neuron_forward(perceptron, input);
    
    // calculate loss
    rac_var_sub_inplace(cost, yhat, target); // reuse 'cost' variable by inplacing a new value

    // backward
    rac_neuron_zero_grad(perceptron);
    rac_var_backward(cost);

    // update
    rac_neuron_update(perceptron, lr * loss->data);
}

// free automatically
vt_mallocator_destroy(alloctr);

// or free manually
rac_var_free(cost);
rac_var_free(target);
rac_neuron_free(perceptron);
plist_var_free(input);
```

For more details check out [`tests/src/main.c`](tests/src/main.c).

## LICENSE
All code is licensed under the BSL license.

