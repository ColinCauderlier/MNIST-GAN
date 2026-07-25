#include "gan.h"


int main()
{
    t_gen   generator;
    t_disc  discriminator;

    init(&generator, &discriminator);

    

    free_all(&generator, &discriminator);
    return (0);
}