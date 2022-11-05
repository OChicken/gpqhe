/*
 * Bootstrap.
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
#include "fhe.h"
#include <complex.h> /* creal, cimag, I */
#include <math.h>

BEGIN_DECLS

/* poly.h */
extern struct poly_ctx polyctx;

/* fhe.h */
extern struct he_ctx hectx;

/* types.c */
extern void double_to_mpi(MPI *r, long double a);
extern double mpi_to_double(MPI a);

/* canemb.c */
void canemb(_Complex double a[], const unsigned int slots);
void invcanemb(_Complex double a[], const unsigned int slots);

/* ntt.c */
extern void ntt(uint64_t a[], const struct rns_ctx *rns);

/* rns.c */
extern void rns_decompose(uint64_t ahat[], const MPI a[], const struct rns_ctx *rns);

struct bootstrap_ctx bootstrapctx;

static inline unsigned int gcd(unsigned int a, unsigned int b) {
  while (b != 0) {
    a %= b;
    a ^= b;
    b ^= a;
    a ^= b;
  }
  return a;
}

static inline void blas_dzlrot(_Complex double vals[], const unsigned int n, const unsigned int rot) {
  unsigned int rem = rot % n;
  if (rem) {
    unsigned int divisor = gcd(rem, n);
    unsigned int gap = n/divisor;
    for (unsigned int i=0; i<divisor; i++) {
      _Complex double tmp = vals[i];
      unsigned int k = i;
      for (unsigned int j=0; j<gap-1; j++) {
        vals[k] = vals[(k+rem)%n];
        k = (k+rem)%n;
      }
      vals[k] = tmp;
    }
  }
}

static inline void blas_dzrrot(_Complex double vals[], const unsigned int n, const unsigned int rot) {
  unsigned int rem = rot%n;
  rem = (n-rem)%n;
  blas_dzlrot(vals, n, rem);
}

static inline unsigned int maxbits(MPI a[], const unsigned int n)
{
  unsigned int norm=0;
  for (unsigned int i=0; i<n; i++) {
    unsigned int tmp = mpi_get_nbits(a[i]);
    norm = (norm>tmp)? norm : tmp;
  }
  return norm;
}

void he_bootstrapctx_init()
{
  /* local variables */
  unsigned int n = polyctx.n;
  unsigned int m = polyctx.m;
  unsigned int nh = polyctx.n/2;
  unsigned int  slots = hectx.slots;
  unsigned int dslots = slots*2;
  unsigned int  gap = nh/slots;
  unsigned int dgap = gap >> 1;
  unsigned int logqL = mpi_get_nbits(hectx.q[hectx.L]);
  unsigned int dim;
  struct rns_ctx *rns = polyctx.rns;
  /* k=sqrt(slots), smaller one. e.g. slots=8, then k=2 */
  unsigned int k = 1<<((unsigned int)log2(slots)>>1);
  _Complex double pvals[slots*2];
  /* alloc */
  //bootstrapctx.rp = malloc(slots*sizeof(uint64_t *));
  bootstrapctx.rp = malloc(slots*sizeof(poly_rns_t));
  bootstrapctx.rpinv = malloc(slots*sizeof(uint64_t *));
  bootstrapctx.bnd    = malloc(slots*sizeof(unsigned int));
  bootstrapctx.bndinv = malloc(slots*sizeof(unsigned int));

  poly_mpi_t pvec;
  poly_mpi_alloc(&pvec);

  for (unsigned int ki=0; ki<slots; ki += k) {
    for (unsigned int pos=ki; pos<ki+k; pos++) {
      for (unsigned int i=0; i<slots-pos; i++) {
        unsigned int deg = ((m - polyctx.ring.cyc_group[i+pos])*i*gap)%m;
        pvals[i] = polyctx.ring.zetas[deg];
        pvals[i+slots] = -cimag(pvals[i]) + I*creal(pvals[i]);
      }
      for (unsigned int i=slots-pos; i<slots; i++) {
        unsigned int deg = ((m - polyctx.ring.cyc_group[i+pos-slots])*i*gap)%m;
        pvals[i] = polyctx.ring.zetas[deg];
        pvals[i+slots] = -cimag(pvals[i]) + I*creal(pvals[i]);
      }
      blas_dzrrot(pvals, dslots, ki);
      invcanemb(pvals, dslots);
      for (unsigned int i=0, j=0; i<dslots; i++, j+=dgap) {
        double_to_mpi(&pvec.coeffs[j   ], round((long double)creal(pvals[i])*hectx.Delta));
        double_to_mpi(&pvec.coeffs[j+nh], round((long double)cimag(pvals[i])*hectx.Delta));
      }
      bootstrapctx.bnd[pos] = maxbits(pvec.coeffs, n);
      dim = (bootstrapctx.bnd[pos] + logqL)/GPQHE_LOGP+1;
      //np = ceil((bndvec[pos] + logQ + 2 * logN + 2)/(double)pbnd);
      poly_rns_alloc(&bootstrapctx.rp[pos], dim);
      rns = polyctx.rns;
      for (unsigned int d=0; d<dim; d++) {
        rns_decompose(&bootstrapctx.rp[pos].coeffs[d*n], pvec.coeffs, rns);
        ntt(&bootstrapctx.rp[pos].coeffs[d*n], rns);
        rns = (d<dim-1)? rns->next : rns;
      }
      for (unsigned int i=0; i<n; i++)
        mpi_set_ui(pvec.coeffs[i], 0);
    }
  }
  // rp1, bnd1
  for (unsigned int i=0; i<slots; ++i) {
    pvals[i] = 0.0;
    pvals[i+slots] = -0.25*I/GPQHE_PI;
  }
  invcanemb(pvals, dslots);
  for (unsigned int i=0, j=0; i<dslots; i++, j+=dgap) {
    double_to_mpi(&pvec.coeffs[j   ], round((long double)creal(pvals[i])*hectx.Delta));
    double_to_mpi(&pvec.coeffs[j+nh], round((long double)creal(pvals[i])*hectx.Delta));
  }
  bootstrapctx.bnd1 = maxbits(pvec.coeffs, n);
  dim = (bootstrapctx.bnd1 + logqL)/GPQHE_LOGP+1;
  poly_rns_alloc(&bootstrapctx.rp1, dim);
  rns = polyctx.rns;
  for (unsigned int d=0; d<dim; d++) {
    rns_decompose(&bootstrapctx.rp1.coeffs[d*n], pvec.coeffs, rns);
    ntt(&bootstrapctx.rp1.coeffs[d*n], rns);
    rns = (d<dim-1)? rns->next : rns;
  }
  for (unsigned int i=0; i<n; i++)
    mpi_set_ui(pvec.coeffs[i], 0);
  // rp2, bnd2
  for (unsigned int i=0; i<slots; ++i) {
    pvals[i] = 0.25/GPQHE_PI;
    pvals[i+slots] = 0.0;
  }
  invcanemb(pvals, dslots);
  for (unsigned int i=0, j=0; i<dslots; i++, j+=dgap) {
    double_to_mpi(&pvec.coeffs[j   ], round((long double)creal(pvals[i])*hectx.Delta));
    double_to_mpi(&pvec.coeffs[j+nh], round((long double)creal(pvals[i])*hectx.Delta));
  }
  bootstrapctx.bnd2 = maxbits(pvec.coeffs, n);
  dim = (bootstrapctx.bnd2 + logqL)/GPQHE_LOGP+1;
  poly_rns_alloc(&bootstrapctx.rp2, dim);
  rns = polyctx.rns;
  for (unsigned int d=0; d<dim; d++) {
    rns_decompose(&bootstrapctx.rp2.coeffs[d*n], pvec.coeffs, rns);
    ntt(&bootstrapctx.rp2.coeffs[d*n], rns);
    rns = (d<dim-1)? rns->next : rns;
  }
  for (unsigned int i=0; i<n; i++)
    mpi_set_ui(pvec.coeffs[i], 0);
  //
  for (unsigned int ki=0; ki<slots; ki+=k) {
    for (unsigned int pos=ki; pos<ki+k; pos++) {
      for (unsigned int i=0; i<slots-pos; i++) {
        unsigned int deg = (polyctx.ring.cyc_group[i]*(i+pos)*gap)%m;
        pvals[i] = polyctx.ring.zetas[deg];
      }
      for (unsigned int i=slots-pos; i<slots; i++) {
        unsigned int deg = (polyctx.ring.cyc_group[i]*(i+pos-slots)*gap)%m;
        pvals[i] = polyctx.ring.zetas[deg];
      }
      blas_dzrrot(pvals, slots, ki);
      invcanemb(pvals, slots);
      for (unsigned int i=0, j=0; i<slots; i++, j+=gap) {
        double_to_mpi(&pvec.coeffs[j   ], round((long double)creal(pvals[i]*hectx.Delta)));
        double_to_mpi(&pvec.coeffs[j+nh], round((long double)cimag(pvals[i]*hectx.Delta)));
      }
      bootstrapctx.bndinv[pos] = maxbits(pvec.coeffs, n);
      dim = (bootstrapctx.bndinv[pos] + logqL)/GPQHE_LOGP+1;
      poly_rns_alloc(&bootstrapctx.rpinv[pos], dim);
      rns = polyctx.rns;
      for (unsigned int d=0; d<dim; d++) {
        rns_decompose(&bootstrapctx.rpinv[pos].coeffs[d*n], pvec.coeffs, rns);
        ntt(&bootstrapctx.rpinv[pos].coeffs[d*n], rns);
        rns = (d<dim-1)? rns->next : rns;
      }
      for (unsigned int i=0; i<n; i++)
        mpi_set_ui(pvec.coeffs[i], 0);
    }
  }
  poly_mpi_free(&pvec);
}

void he_bootstrapctx_exit()
{
  for (unsigned int i=0; i<hectx.slots; i++) {
    poly_rns_free(&bootstrapctx.rp[i]);
    poly_rns_free(&bootstrapctx.rpinv[i]);
  }
  poly_rns_free(&bootstrapctx.rp1);
  poly_rns_free(&bootstrapctx.rp2);
  free(bootstrapctx.rp);
  free(bootstrapctx.rpinv);
  free(bootstrapctx.bnd);
  free(bootstrapctx.bndinv);
}

END_DECLS
