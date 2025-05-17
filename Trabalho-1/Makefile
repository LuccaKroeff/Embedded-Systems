SUBDIRS = CPU GPU

all:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

test:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir test; \
	done