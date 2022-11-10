/*
 * Encoder.
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

#include "poly.h"
#include "gpqhe.h"
#include <complex.h> /* creal, cimag, I */
#include <math.h>    /* round */

BEGIN_DECLS

/* polyctx.h */
extern struct poly_ctx polyctx;

/* fhe.h */
extern struct he_ctx hectx;

/* types.c */
extern void double_to_mpi(MPI *r, long double a);

/* canemb.c */
void canemb(_Complex double a[], const unsigned int slots);
void invcanemb(_Complex double a[], const unsigned int slots);

static inline double blas_dnrmmax(const double x[], const unsigned int len)
{
  double norm = 0;
  for (unsigned int i=0; i<len; i++) {
    double tmp = __builtin_fabs(x[i]);
    if (norm<tmp)
      norm = tmp;
  }
  return norm;
}

/** Encodes complex array into pt, an integral polynomial. */
void encode(poly_mpi_t *r, const _Complex double m[], const double Delta)
{
  _Complex double u[hectx.slots];
  memcpy(u, m, sizeof(u));
  invcanemb(u, hectx.slots);
  unsigned int nh = polyctx.n/2; /* n half */
  unsigned int gap = nh/hectx.slots;
  for (unsigned int i=0, j=0; i<hectx.slots; i++, j+=gap) {
    double_to_mpi(&r->coeffs[j   ], round((long double)creal(u[i])*Delta));
    double_to_mpi(&r->coeffs[j+nh], round((long double)cimag(u[i])*Delta));
  }
}

/** Decodes pt, an integral polynomial, into complex array. */
void decode(_Complex double m[], poly_mpi_t *a, const double Delta)
{
  unsigned int nh = polyctx.n/2;
  unsigned int gap = nh/hectx.slots;
  for (unsigned int i=0, j=0; i<hectx.slots; i++, j+=gap)
    m[i] = mpi_to_double(a->coeffs[j])/Delta + I*mpi_to_double(a->coeffs[j+nh])/Delta;
  canemb(m, hectx.slots);
}

/** canemb_norm - Canonical embedding norm */
double canemb_norm_(const poly_mpi_t *a)
{
  _Complex double m[hectx.slots];
  unsigned int nh = polyctx.n/2;
  unsigned int gap = nh/hectx.slots;
  for (unsigned int i=0, j=0; i<hectx.slots; i++, j+=gap)
    m[i] = mpi_to_double(a->coeffs[j]) + I*mpi_to_double(a->coeffs[j+nh]);
  canemb(m, hectx.slots);
  unsigned int len = hectx.slots*2;
  double u[len];
  for (unsigned int i=0; i<hectx.slots; i++) {
    u[i            ] = round((long double)creal(m[i]));
    u[i+hectx.slots] = round((long double)cimag(m[i]));
  }
  return blas_dnrmmax(u, len);
}

/** canemb_norm - Canonical embedding norm */
double canemb_norm(const _Complex double m[], const double Delta)
{
  unsigned int len = hectx.slots*2;
  double u[len];
  for (unsigned int i=0; i<hectx.slots; i++) {
    u[i            ] = round((long double)creal(m[i])*Delta);
    u[i+hectx.slots] = round((long double)cimag(m[i])*Delta);
  }
  return blas_dnrmmax(u, len);
}

/** Encodes complex array into pt, an integral polynomial. */
void he_ecd(struct he_pt *pt, const _Complex double *m)
{
  pt->nu = hectx.Delta;
  encode(&pt->m, m, pt->nu);
}

/** Decodes pt, an integral polynomial, into complex array. */
void he_dcd(_Complex double *m, struct he_pt *pt)
{
  decode(m, &pt->m, pt->nu);
}

void he_const_pt(he_pt_t *pt, const _Complex double num)
{
  pt->nu = hectx.Delta;
  unsigned int nh = polyctx.n/2;
  double_to_mpi(&pt->m.coeffs[0 ], round((long double)creal(num)*hectx.Delta));
  double_to_mpi(&pt->m.coeffs[nh], round((long double)cimag(num)*hectx.Delta));
}

END_DECLS
