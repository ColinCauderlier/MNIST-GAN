#include "gan.h"

void    forward_layer(t_layer *layer, float *input, t_fcache *cache)
{
    int output_index = 0;
    int input_index;

    while (output_index < layer->output_size)
    {
        input_index = 0;
        while (input_index < layer->input_size)
        {
            cache->sums->z[output_index] += input[input_index] * layer->weights[input_index][output_index];
            input_index++;
        }
        cache->sums->z[output_index] += layer->biases[output_index];
        output_index++;
    }
    output_index = 0;
    while (output_index < layer->output_size)
    {
        cache->sums->a[output_index] = 0;
        if (cache->sums->z[output_index] > 0)
            cache->sums->a[output_index] = cache->sums->z[output_index];
        output_index++;
    }
}