NVCC = nvcc
SRC = binary_search.cu convolution.cu
BIN = $(SRC:.cu=)

all: $(BIN)

%: %.cu
	$(NVCC) -o $@ $<

clean:
	rm -f $(BIN)

test: all
	@echo "Testando GPU..."
	@for bin in $(BIN); do "Executando $$bin:"; ./$$bin; echo ""; done
