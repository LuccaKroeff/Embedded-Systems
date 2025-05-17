CC = gcc
CFLAGS =
SRC = binary_search.c convolution.c
BIN = $(SRC:.c=)

all: $(BIN)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BIN)

test: all
	@echo "Testando executáveis em CPU/..."
	@for bin in $(BIN); do echo "Executando $$bin:"; ./$$bin; echo ""; done

