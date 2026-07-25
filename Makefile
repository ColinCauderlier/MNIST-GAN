CC      = gcc
CFLAGS  = -Wall -Wextra -Werror -lm
NAME    = gan
SRCS    = main.c init.c init_layer.c free_layer_utils.c free.c
OBJS    = $(SRCS:.c=.o)

all: $(NAME)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(NAME)

clean:
	rm -f $(OBJS)

fclean: clean
	rm -f $(NAME)

re: fclean all