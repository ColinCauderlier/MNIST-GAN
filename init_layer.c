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
This is an initialisation of a weight of a hidden layer, the He initialisation, ideal for ReLU and LeakyReLU
(float)rand() / RAND_MAX gives a random nuber between 0 and 1
- 0.5 centers this number around 0, the range at this moment is [-0.5, 0.5]
* sqrt(2 / hidden->nb_inputs) gives us a random number in the range [-0.0505, 0.0505]
*/
float   get_random_weight_He(t_layer *layer)
{
    unsigned int random_int = rand();
    float weight = (((float) random_int / RAND_MAX) - 0.5) * sqrt(2.0 / layer->input_size);
    return (weight);
}

/*
This is an initialisation of a weight of the output layer, the Xavier Glorot initialisation, ideal for sigmoids
(float)rand() / RAND_MAX gives a random nuber between 0 and 1
- 0.5 centers this number around 0, the range at this moment is [-0.5, 0.5]
* sqrt(2 / output->nb_inputs + output->nb_outputs) gives us a random number in the range [-0.088, 0.088]
*/
float   get_random_weight_Glorot(t_layer *layer)
{
    unsigned int random_int = rand();
    float weight = (((float) random_int / RAND_MAX) - 0.5) * sqrt(2.0 / (layer->input_size + layer->output_size));
    return (weight);
}

/*
Initialize the weights and biases of a full layer using the Glorot initialization
*/
void    init_weights_Glorot(t_layer *layer)
{
    int input_index = 0;
    int output_index = 0;

    while (input_index < layer->input_size)
    {
        output_index = 0;
        while(output_index < layer->output_size)
        {
            layer->weights[input_index][output_index] = get_random_weight_Glorot(layer);
            layer->weights_grad[input_index][output_index] = 0;
            output_index++;
        }
        input_index++;
    }
    output_index = 0;
    while (output_index < layer->output_size)
    {
        layer->biases[output_index] = 0;
        layer->biases_grad[output_index] = 0;
        output_index++;
    }
}
/*
Initialize the weights and biases of a full layer using the He initialization
*/
void    init_weights_He(t_layer *layer)
{
    int input_index = 0;
    int output_index = 0;

    while (input_index < layer->input_size)
    {
        output_index = 0;
        while(output_index < layer->output_size)
        {
            layer->weights[input_index][output_index] = get_random_weight_He(layer);
            layer->weights_grad[input_index][output_index] = 0;
            output_index++;
        }
        input_index++;
    }
    output_index = 0;
    while (output_index < layer->output_size)
    {
        layer->biases[output_index] = 0;
        layer->biases_grad[output_index] = 0;
        output_index++;
    }
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