CC = gcc
CFLAGS =
SRC = sudoku.c convolution.c
BIN = $(SRC:.c=)

all: $(BIN)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BIN)

test: all
	@echo "\n================================"
	@echo "Testando executáveis em CPU/..."
	@for bin in $(BIN); do echo "\nExecutando $$bin:"; ./$$bin; done
	@echo "================================\n"

