#include "gan.h"

/*
Fully allocate a layer weights, freeing everything and returning -1 if a problem occurs
*/
static int     allocate_layer_weights(t_layer *layer)
{
    int input_index;

    layer->weights = malloc(layer->input_size * sizeof(int*));
    if (!layer->weights)
        return (-1);
    layer->weights_grad = malloc(layer->input_size * sizeof(int*));
    if (!layer->weights_grad)
    {
        free_weights_gradients(layer);
        return (-1);
    }
    input_index = 0;
    while (input_index < layer->input_size)
    {
        layer->weights[input_index] = malloc(layer->output_size * sizeof(int));
        if (!layer->weights[input_index])
        {
            free_weights_error(layer, input_index);
            free_weights_gradients_error(layer, input_index - 1);
            return (-1);
        }
        layer->weights_grad[input_index] = malloc(layer->output_size * sizeof(int));
        if (!layer->weights_grad[input_index])
        {
            free_weights_error(layer, input_index);
            free_weights_gradients_error(layer, input_index);
            return (-1);
        }
        input_index++;
    }
    return (0);
}

/*
Fully allocate a layer biases, freeing everything and returning -1 if a problem occurs
*/
static int     allocate_layer_biases(t_layer *layer)
{
    layer->biases = malloc(layer->output_size * sizeof(int));
    if (!layer->biases)
        return (-1);   
    layer->biases_grad = malloc(layer->output_size * sizeof(int));
    if (!layer->biases_grad)
    {
        free_biases(layer);
        return (-1);   
    }
    return (0);
}

/*
Fully initialize a layer of neurons, returning -1 if a problem occurs
*/
int init_layer(t_layer *layer, const t_method method)
{
    if (allocate_layer_weights(layer) == -1)
        return (-1);
    if (allocate_layer_biases(layer) == -1)
    {
        free_weights(layer);
        return (-1);
    }
    if (method == HE)
        init_weights_He(layer);
    else if (method == GLOROT)
        init_weights_Glorot(layer);
    return (0);
}