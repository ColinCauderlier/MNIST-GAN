#include "gan.h"

/*
Entirely frees the weights of a layer
*/
void    free_weights(t_layer *layer)
{
    int input = 0;
    int output;

    while (input < layer->input_size)
    {
        free(layer->weights[input]);
        input++;
    }
    free(layer->weights);
}

void    free_weights_gradients(t_layer *layer)
{
    int input = 0;
    int output;

    while (input < layer->input_size)
    {
        free(layer->weights_grad[input]);
        input++;
    }
    free(layer->weights_grad);
}


/*
Fully initialize a layer of neurons, freeing everything and returning -1 if a problem occurs
*/
int     init_layer(t_layer *layer)
{
    layer->weights = malloc(layer->input_size * sizeof(int*));
    if (!layer->weights)
        return (-1);



    return (0);
}

static void    init_generator_sizes(t_gen *generator)
{
    generator->hidden1_layer.input_size = GEN_NOISE_INPUT;
    generator->hidden1_layer.output_size = GEN_HIDDEN1_NEURONS;
    generator->hidden2_layer.input_size = GEN_HIDDEN1_NEURONS;
    generator->hidden2_layer.output_size = GEN_HIDDEN2_NEURONS;
    generator->output_layer.input_size = GEN_HIDDEN2_NEURONS;
    generator->output_layer.output_size = GEN_OUTPUT_NEURONS;
}

int     init_generator(t_gen *generator)
{
    init_generator_sizes(generator);

    return (0);
}

int    init_discriminator(t_disc *discriminator)
{

    return (0);
}

int    init(t_gen *generator, t_disc *discriminator)
{
    if (init_generator(generator) == -1)
        return (-1);
    if (init_discriminator(discriminator) == -1)
        return (-1);
    return (0);
}