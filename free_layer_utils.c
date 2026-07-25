#include "gan.h"

/*
Entirely frees the weights of a layer
*/
void    free_weights(t_layer *layer)
{
    int input = 0;

    while (input < layer->input_size)
    {
        free(layer->weights[input]);
        input++;
    }
    free(layer->weights);
}

/*
Entirely frees the weights gradients of a layer
*/
void    free_weights_gradients(t_layer *layer)
{
    int input = 0;

    while (input < layer->input_size)
    {
        free(layer->weights_grad[input]);
        input++;
    }
    free(layer->weights_grad);
}

/*
Entirely frees the biases of a layer
*/
void    free_biases(t_layer *layer)
{
    free(layer->biases);
}

/*
Entirely frees the biases gradients of a layer
*/
void    free_biases_gradients(t_layer *layer)
{
    free(layer->biases_grad);
}

/*
Frees the weights of a layer from index 0 to input_index excluded
*/
void    free_weights_error(t_layer *layer, int input_index)
{
    int input = 0;

    while (input < input_index)
    {
        free(layer->weights[input]);
        input++;
    }
    free(layer->weights);
}

/*
Frees the weights gradients of a layer from index 0 to input_index excluded
*/
void    free_weights_gradients_error(t_layer *layer, int input_index)
{
    int input = 0;

    while (input < input_index)
    {
        free(layer->weights_grad[input]);
        input++;
    }
    free(layer->weights);
}