#ifndef GAN_H
# define GAN_H

/*
INCLUDES
*/

# include <stdlib.h>
# include <fcntl.h>
# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <time.h>

/*
CONSTANTS
*/

# define LEARNING_RATE 0.05
# define NB_EPOCHS 20
# define NB_IMAGES_TRAINING_SET 60000
# define NB_IMAGES_TEST_SET 10000
# define IMAGE_SIZE 784

//Discriminator model
# define DISC_HIDDEN1_NEURONS 256
# define DISC_HIDDE21_NEURONS 128
# define DISC_OUTPUT_NEURONS 1

//Generative model
# define GEN_NOISE_INPUT 100
# define GEN_HIDDEN1_NEURONS 128
# define GEN_HIDDEN2_NEURONS 256
# define GEN_OUTPUT_NEURONS 784

//Paths to the dataset
# define PATH_TRAINING_LABELS "../train-labels-idx1-ubyte/train-labels.idx1-ubyte"
# define PATH_TRAINING_IMAGES "../train-images-idx3-ubyte/train-images.idx3-ubyte"
# define PATH_TEST_LABELS "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
# define PATH_TEST_IMAGES "../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"

/*
STRUCTURES
*/

typedef struct s_image
{
	float	        data[IMAGE_SIZE];
    int             label;
    int             type; //0 is fake, 1 is real
}					t_image;

typedef struct s_layer
{
    int     input_size;
    int     output_size;

    float   **weights;
    float   *biases;

    float   **weights_grad;
    float   *biases_grad;
}                   t_layer;

typedef struct s_generator
{
    t_layer     hidden1_layer; //100 -> 128
    t_layer     hidden2_layer; //128 -> 256
    t_layer     output_layer;  //256 -> 784
}                   t_gen;

typedef struct s_discriminator
{
    t_layer     hidden1_layer; //784 -> 256
    t_layer     hidden2_layer; //256 -> 128
    t_layer     output_layer;  //128 -> 1
}                   t_disc;

/*
FUNCTIONS
*/

//init.c
int    init(t_gen *generator, t_disc *discriminator);
int    init_discriminator(t_disc *discriminator);
int    init_generator(t_gen *generator);
int     init_layer(t_layer *layer);

//free_layer_utils.c
void    free_weights(t_layer *layer);
void    free_weights_gradients(t_layer *layer);
void    free_biases(t_layer *layer);
void    free_biases_gradients(t_layer *layer);

#endif