/*
 * Types definitions and conversions.
 * Copyright (C) shouran.ma@rwth-aachen.de
 *
 * This file is part of GPQHE.
 *
 * GPQHE is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * GPQHE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "types.h"
#include <assert.h>
#include <errno.h>

BEGIN_DECLS

extern MPI GPQHE_TWO;

uint64_t mpi_to_u64(MPI a)
{
  uint64_t num;
  size_t length = mpi_get_nbits (a);
  /* if is zero */
  if (!length)
    return 0;
  /* if not zero, and pos */
  assert(mpi_is_neg(a)==0);
  num = mpi_test_bit(a, length);
  while (length-- > 0){
    num <<= 1;
    num += mpi_test_bit(a, length);
  }
  return num;
}

int64_t mpi_to_s64(MPI a)
{
  int64_t num;
  size_t length = gcry_mpi_get_nbits (a);
  /* if is zero */
  if (!length)
    return 0;
  /* if not zero, and neg */
  if (gcry_mpi_is_neg(a)) {
    MPI b = gcry_mpi_new(0);
    gcry_mpi_set(b,a);
    gcry_mpi_neg(b,b);
    num = gcry_mpi_test_bit(b, length);
    while (length-- > 0) {
      num <<= 1;
      num += gcry_mpi_test_bit(b, length);
    }
    num = -num;
    mpi_release(b);
  }
  /* if not zero, and pos */
  else {
    num = gcry_mpi_test_bit(a, length);
    while (length-- > 0) {
      num <<= 1;
      num += gcry_mpi_test_bit(a, length);
    }
  }
  return num;
}

double mpi_to_double(MPI a)
{
  double num;
  size_t length = mpi_get_nbits (a);
  /* if is zero */
  if (!length)
    return 0;
  /* if not zero, and neg */
  if (mpi_is_neg(a)) {
    MPI b = mpi_new(0);
    mpi_set(b,a);
    mpi_neg(b,b);
    num = mpi_test_bit(b, length);
    while (length-- > 0) {
      num *= 2.;
      num += mpi_test_bit(b, length);
    }
    num = -num;
    mpi_release(b);
  }
  /* if not zero, and pos */
  else {
    num = mpi_test_bit(a, length);
    while (length-- > 0) {
      num *= 2.;
      num += mpi_test_bit(a, length);
    }
  }
  return num;
}

void mpi_smod(MPI r, const MPI q, const MPI qh)
{
  mpi_mod(r, r, q);
  if (mpi_cmp(r, qh)>=0)
    mpi_sub(r, r, q);
}

void mpi_rdiv(MPI q, const MPI a, const MPI m)
{
  assert(!mpi_is_neg(m));
  MPI mh = mpi_new(0);
  MPI r  = gcry_mpi_new(0);
  mpi_fdiv(mh, NULL, m, GPQHE_TWO); /* mh = floor(m/2) */
  /* main */
  mpi_fdiv(q, r, a, m);
  if (mpi_cmp(r, mh)>0)
    mpi_add_ui(q, q, 1);
  /* release */
  mpi_release(mh);
  mpi_release(r);
}

uint64_t loadu64_littleendian(const uint8_t x[8])
{
  uint64_t r;
  r  = (uint64_t)x[0];
  r |= (uint64_t)x[1] <<  8;
  r |= (uint64_t)x[2] << 16;
  r |= (uint64_t)x[3] << 24;
  r |= (uint64_t)x[4] << 32;
  r |= (uint64_t)x[5] << 40;
  r |= (uint64_t)x[6] << 48;
  r |= (uint64_t)x[7] << 56;
  return r;
}

uint64_t loadnbits_littleendian(const uint8_t buf[], const unsigned int nbits)
{
  if (nbits>64) {
    errno = EINVAL;
    fprintf(stderr, "\033[1m\033[31merror:\033[0m \033[1m%s\033[0m. "
      "Must guarantee nbits<=64.\n", strerror(errno));
    abort();
  }
  if (nbits==64)
    return loadu64_littleendian(buf);
  uint64_t a = 0;
  unsigned int q = nbits/8; /* q<=7 is guaranteed, since nbits<64. */
  unsigned int r = nbits%8;
  for (unsigned int i=0; i<q; i++)
    a |= buf[i]<<(8*i);
  uint8_t x = 0;
  for (unsigned int i=0; i<r; i++)
    x |= ((buf[q]>>i)&0x01)<<i;
  a |= x<<(8*q);
  return a;
}

void loadmpi_littleendian(MPI a, const uint8_t buf[], const unsigned int nbits)
{
  mpi_set_ui(a, 0);
  MPI shift = mpi_new(0);
  unsigned int q = nbits/8;
  unsigned int r = nbits%8;
  for (unsigned int i=0; i<q; i++) {
    mpi_set_ui(shift, buf[i]);
    mpi_lshift(shift, shift, 8*i);
    mpi_add(a, a, shift);
  }
  uint8_t x=0;
  for (unsigned int j=0; j<r; j++)
    x |= ((buf[q]>>j)&0x01)<<j;
  mpi_set_ui(shift, x);
  mpi_lshift(shift, shift, 8*q);
  mpi_add(a, a, shift);
  mpi_release(shift);
}

void show_mpi (MPI a)
{
  gcry_error_t err = GPG_ERR_NO_ERROR;
  gcry_sexp_t data;
  char *buf;
  size_t size;
  err = gcry_sexp_build(&data, NULL, "%m", a);
  if (err)
    fprintf(stderr, "Error in %s.", __func__);
  size = gcry_sexp_sprint (data, GCRYSEXP_FMT_ADVANCED, NULL, 0);
  buf = (char *)malloc (size);
  gcry_sexp_sprint (data, GCRYSEXP_FMT_ADVANCED, buf, size);
  fflush(stdout);
  fprintf (stderr, "%s", buf);
  free (buf);
  gcry_sexp_release(data);
}

#if 0
void show_mpi_(MPI a)
{
  size_t length = mpi_get_nbits (a);
  unsigned int limbs = length/BITS_PER_LONG+1;
  uint64_t *b = malloc(limbs*sizeof(uint64_t));
  for (unsigned int i=0; i<limbs; i++) {
    b[i] = 0;
    if (i<limbs-1)
      for (unsigned int j=0; j<BITS_PER_LONG; j++)
        b[i] += (uint64_t)mpi_test_bit(a,i*BITS_PER_LONG+j)<<j;
    else
      for (unsigned int j=0; j<length%BITS_PER_LONG; j++)
        b[i] += (uint64_t)mpi_test_bit(a,i*BITS_PER_LONG+j)<<j;
  }
  for (unsigned int i=0; i<limbs; i++)
    printf("%lx\n", b[i]);
  free(b);
}
#endif

void double_to_mpi(MPI *r, long double a)
{
  int sign;
  if (a>=0)
    sign = 1;
  else
    sign = -1, a=-a;
  long double uint64_max = UINT64_MAX;
  size_t count = a/uint64_max;
  a -= count*uint64_max;
  mpi_set_ui(*r, (uint64_t)a);
  if (count) {
    MPI shift = mpi_new(0);
    mpi_set_ui(shift, UINT64_MAX);
    mpi_mul_ui(shift, shift, count);
    mpi_add(*r, *r, shift);
    mpi_release(shift);
  }
  if (sign==-1)
    mpi_neg(*r,*r);
}

END_DECLS
