#include "gan.h"

/*
Entirely frees a layer, the layer itself is not allocated only the arrays
*/
void    free_layer(t_layer *layer)
{
    free_weights(layer);
    free_weights_gradients(layer);
    free_biases(layer);
    free_biases_gradients(layer);
}

/*
Entirely frees the generator's layers
*/
void    free_generator(t_gen *generator)
{
    free_layer(&generator->hidden1_layer);
    free_layer(&generator->hidden2_layer);
    free_layer(&generator->output_layer);
}

/*
Entirely frees the discriminator's layers
*/
void    free_discriminator(t_disc *discriminator)
{
    free_layer(&discriminator->hidden1_layer);
    free_layer(&discriminator->hidden2_layer);
    free_layer(&discriminator->output_layer);
}

void    free_all(t_gen *generator, t_disc *discriminator)
{
    free_generator(generator);
    free_discriminator(discriminator);
}