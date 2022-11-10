ROOT ?= ..

HEADERS = \
	config.h \
	types.h \
	params.h \
	poly.h \
	fips202.h \
	kem.h \
	symmetric.h \
	gpqhe.h
SOURCES = \
	types.c \
	precomp.c \
	reduce.c \
	ntt.c \
	rns.c \
	poly.c \
	fips202.c \
	canemb.c \
	rng.c \
	symmetric-shake.c \
	sample.c \
	he-mem.c \
	he-info.c \
	he-encode.c \
	he-encrypt.c \
	he-rescale.c \
	he-kem.c \
	he-add.c \
	he-mult.c \
	he-automorphism.c \
	he-algo.c \
	he-bootstrap.c

libgpqhe.so: $(HEADERS) $(SOURCES)
#	$(CC) $(CFLAGS) -DRANDOM   $(SOURCES) -fPIC -shared -lgcrypt -lm -o $(TARGET)/$@
#	$(CC) $(CFLAGS) -DAES256   $(SOURCES) -fPIC -shared -lgcrypt -lm -o $(TARGET)/$@
	$(CC) $(CFLAGS) -DSUPERCOP $(SOURCES) -fPIC -shared -lgcrypt -lm -o $(LIBDIR)/$@

include $(ROOT)/Makefile
