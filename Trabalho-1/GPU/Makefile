NVCC = nvcc
SRC = binary_search.cu convolution.cu
BIN = $(SRC:.cu=)

all: $(BIN)

%: %.cu
	$(NVCC) -o $@ $<

clean:
	rm -f $(BIN)

test: all
	@echo "\n================================"
	@echo "Testando executáveis em GPU/..."
	@for bin in $(BIN); do echo "\nExecutando $$bin:"; ./$$bin; done
	@echo "================================\n"