#!/usr/bin/make -f
#
# Copyright (C) shouran.ma@rwth-aachen.de
#
# This file is part of GPQHE.
#
# GPQHE is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; either version 2.1 of
# the License, or (at your option) any later version.
#
# GPQHE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, see <http://www.gnu.org/licenses/>.

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
