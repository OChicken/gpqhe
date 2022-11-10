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

ROOT = ..
CFLAGS  += -I$(ROOT)/src -I$(ROOT)/pmu
LDFLAGS ?= -L${ROOT}/lib

test-crt: crt.c $(ROOT)/src/$(HEADERS) $(ROOT)/src/$(SOURCES)
	cd $(LIBDIR) && mv libgpqhe.so libgpqhe_.so
	cd $(ROOT)/src && $(CC) $(CFLAGS) -DTEST_CRT -DSUPERCOP $(SOURCES) -fPIC -shared -lgcrypt -lm -o $(LIBDIR)/libgpqhe.so
	$(CC) $(CFLAGS) crt.c     $(LDFLAGS) -lgpqhe -lpmu -lgcrypt -o $@
	./$@
	cd $(LIBDIR) && mv libgpqhe_.so libgpqhe.so
	rm $@

test-polymul: polymul.c
	$(CC) $(CFLAGS) polymul.c $(LDFLAGS) -lgpqhe -lpmu -lgcrypt -o $@

test-gpqhe: gpqhe.c
	$(CC) $(CFLAGS) gpqhe.c   $(LDFLAGS) -lgpqhe -lpmu -lgcrypt -lm -o $@

test-misc: misc.c
	$(CC) $(CFLAGS) misc.c    $(LDFLAGS) -lgpqhe -lpmu -lgcrypt -lm -o $@

test-rng: rng.c
	$(CC) $(CFLAGS) rng.c     $(LDFLAGS) -lgpqhe -lpmu -lgcrypt -lm -o $@

include $(ROOT)/src/Makefile

clean:
	rm test-*
