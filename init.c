#include "gan.h"

static void    init_generator_sizes(t_gen *generator)
{
    generator->hidden1_layer.input_size = GEN_NOISE_INPUT;
    generator->hidden1_layer.output_size = GEN_HIDDEN1_NEURONS;
    generator->hidden2_layer.input_size = GEN_HIDDEN1_NEURONS;
    generator->hidden2_layer.output_size = GEN_HIDDEN2_NEURONS;
    generator->output_layer.input_size = GEN_HIDDEN2_NEURONS;
    generator->output_layer.output_size = GEN_OUTPUT_NEURONS;
}

static void    init_discriminator_sizes(t_disc *discriminator)
{
    discriminator->hidden1_layer.input_size = IMAGE_SIZE;
    discriminator->hidden1_layer.output_size = DISC_HIDDEN1_NEURONS;
    discriminator->hidden2_layer.input_size = DISC_HIDDEN1_NEURONS;
    discriminator->hidden2_layer.output_size = DISC_HIDDEN2_NEURONS;
    discriminator->output_layer.input_size = DISC_HIDDEN2_NEURONS;
    discriminator->output_layer.output_size = DISC_OUTPUT_NEURONS;
}

static int     init_generator(t_gen *generator)
{
    init_generator_sizes(generator);
    init_layer(&generator->hidden1_layer, HE);
    init_layer(&generator->hidden2_layer, HE);
    init_layer(&generator->output_layer, GLOROT);
    return (0);
}

static int    init_discriminator(t_disc *discriminator)
{
    init_discriminator_sizes(discriminator);
    init_layer(&discriminator->hidden1_layer, HE);
    init_layer(&discriminator->hidden2_layer, HE);
    init_layer(&discriminator->output_layer, GLOROT);
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