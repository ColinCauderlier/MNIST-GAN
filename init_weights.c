#include "gan.h"

/*
This is an initialisation of a weight of a hidden layer, the He initialisation, ideal for ReLU and LeakyReLU
(float)rand() / RAND_MAX gives a random nuber between 0 and 1
- 0.5 centers this number around 0, the range at this moment is [-0.5, 0.5]
* sqrt(2 / hidden->nb_inputs) gives us a random number in the range [-0.0505, 0.0505]
*/
static float   get_random_weight_He(t_layer *layer)
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
static float   get_random_weight_Glorot(t_layer *layer)
{
    unsigned int random_int = rand();
    float weight = (((float) random_int / RAND_MAX) - 0.5) * sqrt(2.0 / (layer->input_size + layer->output_size));
    return (weight);
}

/*
Initialize the weights and biases of a full layer using the Glorot initialization
*/
static void    init_weights_Glorot(t_layer *layer)
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
static void    init_weights_He(t_layer *layer)
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