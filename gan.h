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
# define DISC_HIDDEN2_NEURONS 128
# define DISC_OUTPUT_NEURONS 1
# define DISC_NB_LAYERS 3

//Generative model
# define GEN_NOISE_INPUT 100
# define GEN_HIDDEN1_NEURONS 128
# define GEN_HIDDEN2_NEURONS 256
# define GEN_OUTPUT_NEURONS 784
# define GEN_NB_LAYERS 3

//Paths to the dataset
# define PATH_TRAINING_LABELS "../train-labels-idx1-ubyte/train-labels.idx1-ubyte"
# define PATH_TRAINING_IMAGES "../train-images-idx3-ubyte/train-images.idx3-ubyte"
# define PATH_TEST_LABELS "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
# define PATH_TEST_IMAGES "../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"

/*
STRUCTURES
*/

typedef enum s_method
{
    HE,
    GLOROT,
}					t_method;

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
    t_fcache    cache;
}                   t_gen;

typedef struct s_discriminator
{
    t_layer     hidden1_layer; //784 -> 256
    t_layer     hidden2_layer; //256 -> 128
    t_layer     output_layer;  //128 -> 1
    t_fcache    cache;
}                   t_disc;

typedef struct s_layer_cache
{
    float   *z;
    float   *a;
}					t_lcache;

//Dynamic number of layers
typedef struct s_forward_cache
{
    t_lcache    *sums;
    int         nb_layers;
    float       loss;
}					t_fcache;

/*
FUNCTIONS
*/

//init.c
int    init(t_gen *generator, t_disc *discriminator);

//init_layer.c
int    init_layer(t_layer *layer, const t_method method);

//init_weights.c
void    init_weights_He(t_layer *layer);
void    init_weights_Glorot(t_layer *layer);

//free_layer_utils.c
void    free_weights(t_layer *layer);
void    free_weights_gradients(t_layer *layer);
void    free_biases(t_layer *layer);
void    free_biases_gradients(t_layer *layer);
void    free_weights_error(t_layer *layer, int input_index);
void    free_weights_gradients_error(t_layer *layer, int input_index);

//free.c
void    free_layer(t_layer *layer);
void    free_generator(t_gen *generator);
void    free_discriminator(t_disc *discriminator);
void    free_all(t_gen *generator, t_disc *discriminator);

#endif