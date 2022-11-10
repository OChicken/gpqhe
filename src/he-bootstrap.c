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
#include "gpqhe.h"
#include <complex.h> /* creal, cimag, I */
#include <math.h>

BEGIN_DECLS

/* poly.h */
extern struct poly_ctx polyctx;

/* fhe.h */
extern struct he_ctx hectx;

/* types.c */
extern void double_to_mpi(MPI *r, long double a);

/* canemb.c */
void canemb(_Complex double a[], const unsigned int slots);
void invcanemb(_Complex double a[], const unsigned int slots);

/* ntt.c */
extern void ntt(uint64_t a[], const struct rns_ctx *rns);

/* rns.c */
extern void rns_decompose(uint64_t ahat[], const MPI a[], const struct rns_ctx *rns);

struct bootstrap_ctx bootstrapctx;

static inline unsigned int gcd(unsigned int a, unsigned int b)
{
  while (b != 0) {
    a %= b;
    a ^= b;
    b ^= a;
    a ^= b;
  }
  return a;
}

static inline void blas_dzlrot(_Complex double vals[], const unsigned int n, const unsigned int rot)
{
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

static inline void blas_dzrrot(_Complex double vals[], const unsigned int n, const unsigned int rot)
{
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

#if 0
void he_bootstrapctx_init()
{
  /* local variables */
  double Delta = mpi_to_double(hectx.q[hectx.L]);
  bootstrapctx.Delta = Delta;
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
  /* n2*n1=slots, n1 is giant step, n2 is baby step. E.g. slots=8, then n2=2, n1=4 */
  unsigned int n2 = 1<<((unsigned int)log2(slots)>>1);
  _Complex double pvals[slots*2];
  /* alloc */
  //bootstrapctx.rp = malloc(slots*sizeof(uint64_t *));
  bootstrapctx.rp = malloc(slots*sizeof(poly_rns_t));
  bootstrapctx.rpinv = malloc(slots*sizeof(uint64_t *));
  bootstrapctx.bnd    = malloc(slots*sizeof(unsigned int));
  bootstrapctx.bndinv = malloc(slots*sizeof(unsigned int));

  poly_mpi_t pvec;
  poly_mpi_alloc(&pvec);

  for (unsigned int ki=0; ki<slots; ki += n2) {
    for (unsigned int pos=ki; pos<ki+n2; pos++) {
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
        double_to_mpi(&pvec.coeffs[j   ], round((long double)creal(pvals[i])*Delta));
        double_to_mpi(&pvec.coeffs[j+nh], round((long double)cimag(pvals[i])*Delta));
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
    double_to_mpi(&pvec.coeffs[j   ], round((long double)creal(pvals[i])*Delta));
    double_to_mpi(&pvec.coeffs[j+nh], round((long double)creal(pvals[i])*Delta));
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
    double_to_mpi(&pvec.coeffs[j   ], round((long double)creal(pvals[i])*Delta));
    double_to_mpi(&pvec.coeffs[j+nh], round((long double)creal(pvals[i])*Delta));
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
  for (unsigned int ki=0; ki<slots; ki+=n2) {
    for (unsigned int pos=ki; pos<ki+n2; pos++) {
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
        double_to_mpi(&pvec.coeffs[j   ], round((long double)creal(pvals[i]*Delta)));
        double_to_mpi(&pvec.coeffs[j+nh], round((long double)cimag(pvals[i]*Delta)));
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

static void normalize(he_ct_t *ct)
{
  MPI q = mpi_copy(hectx.q[ct->l]);
  for (unsigned int i=0; i<polyctx.n; i++) {
    if (mpi_get_nbits(ct->c0.coeffs[i])==mpi_get_nbits(q))
      mpi_sub(ct->c0.coeffs[i], ct->c0.coeffs[i], q);
    if (mpi_get_nbits(ct->c1.coeffs[i])==mpi_get_nbits(q))
      mpi_sub(ct->c1.coeffs[i], ct->c1.coeffs[i], q);
  }
  mpi_release(q);
}

void he_coeff2slot(he_ct_t *ct, const he_evk_t *rk)
{
  /* local variables */
  unsigned int n = polyctx.n;
  unsigned int slots = hectx.slots;
  unsigned int n2 = 1<<((unsigned int)log2(slots)>>1);
  MPI q  = mpi_copy(hectx.q [ct->l]);
  he_ct_t ct_rot[n2];
  he_ct_t tmp[n2];
  poly_rns_t ctrothat, ctrotc1hat, tmpc0hat, tmpc1hat;
  poly_rns_alloc(&ctrothat, 1);
  for (unsigned int i=0; i<n2; i++)
    he_alloc_ct(&ct_rot[i]), he_alloc_ct(&tmp[i]);
  he_copy_ct(&ct_rot[0], ct);
  for (unsigned int r=0; r<n2; r++){
    he_rot(&ct_rot[r], r, rk);
    struct rns_ctx *rns = polyctx.rns;
    unsigned int dim = bootstrapctx.bnd[r];
    poly_rns_alloc(&tmpc0hat, dim);
    poly_rns_alloc(&tmpc1hat, dim);
    for (unsigned int d=0; d<dim; d++) {
      rns_decompose(ctrothat.coeffs, ct_rot[r].c0.coeffs, rns);
      ntt(ctrothat.coeffs, rns);
      poly_rns_mul(&tmpc0hat.coeffs[d*n], ctrothat.coeffs, bootstrapctx.rp[r].coeffs, rns);
      rns_decompose(ctrothat.coeffs, ct_rot[r].c0.coeffs, rns);
      ntt(ctrothat.coeffs, rns);
      poly_rns_mul(&tmpc0hat.coeffs[d*n], ctrothat.coeffs, bootstrapctx.rp[r].coeffs, rns);
      rns = (d<dim-1)? rns->next : rns;
    }
    poly_rns2mpi(&tmp[r].c0, &tmpc0hat, rns, q);
    poly_rns2mpi(&tmp[r].c1, &tmpc1hat, rns, q);
    poly_rns_free(&tmpc0hat);
    poly_rns_free(&tmpc1hat);
  }
  for (unsigned int i=0; i<n2; i++)
    he_free_ct(&ct_rot[i]);
  mpi_release(q);
}

#endif

#if 1
void he_bootstrapctx_init()
{
  unsigned int n = polyctx.n;
  unsigned int slots = hectx.slots;
  unsigned int size = slots*slots*sizeof(_Complex double);
  bootstrapctx.U0       = malloc(size);
  bootstrapctx.U1       = malloc(size);
  bootstrapctx.U0_T     = malloc(size);
  bootstrapctx.U1_T     = malloc(size);
  bootstrapctx.U0_conjT = malloc(size);
  bootstrapctx.U1_conjT = malloc(size);
  _Complex double U0[slots*slots];
  _Complex double U1[slots*slots];
  _Complex double U0_T[slots*slots];
  _Complex double U1_T[slots*slots];
  _Complex double U0_conjT[slots*slots];
  _Complex double U1_conjT[slots*slots];
  printf("%s %u\n", __func__, __LINE__);
  unsigned int nh = polyctx.n/2; /* n half */
  unsigned int gap = nh/slots;
  for (unsigned int i=0, ki=0; i<slots; i++, ki+=gap) {
    double theta = 2*GPQHE_PI*polyctx.ring.cyc_group[ki] / polyctx.m;
    _Complex double zeta = cos(theta) + I*sin(theta);
    _Complex double U[n];
    for (unsigned int j=0; j<n; j++)
      U[j] = (j==0)? 1 : U[j-1]*zeta;
    for (unsigned int j=0, k=0; j<slots; j++, k+=gap) {
      U0[i*slots+j] = U[k];
      U1[i*slots+j] = U[k+nh];
    }
    for (unsigned int j=0; j<slots; j++) {
      U0_T[j*slots+i] = U0[i*slots+j];
      U1_T[j*slots+i] = U1[i*slots+j];
      U0_conjT[j*slots+i] = conj(U0_T[j*slots+i]);
      U1_conjT[j*slots+i] = conj(U1_T[j*slots+i]);
    }
  }
  memcpy(bootstrapctx.U0, U0, size);
  memcpy(bootstrapctx.U1, U1, size);
  memcpy(bootstrapctx.U0_T, U0_T, size);
  memcpy(bootstrapctx.U1_T, U1_T, size);
  memcpy(bootstrapctx.U0_conjT, U0_conjT, size);
  memcpy(bootstrapctx.U1_conjT, U1_conjT, size);
}

void he_bootstrapctx_exit()
{
  free(bootstrapctx.U0);
  free(bootstrapctx.U1);
  free(bootstrapctx.U0_T);
  free(bootstrapctx.U1_T);
  free(bootstrapctx.U0_conjT);
  free(bootstrapctx.U1_conjT);
}

/** coefficient to slots
 * Takes an encryption of t(x) = t_0 + t_1x + ... and transforms to encryptions
 * of (t_0, t_1, ..., t_(d/2)) and (t_(d/2 + 1), ..., t_(n-1)) before these
 * vectors are encoded. */
void he_coeff2slot(he_ct_t *ct_real, he_ct_t *ct_imag,
                   const he_ct_t *ct,
                   const he_evk_t *ck, const he_evk_t *rk)
{
#if 0
  COMPLEX U0_ConjTrans[GPQHE_CKKS_SLOT*GPQHE_CKKS_SLOT];
  COMPLEX U0_Trans[GPQHE_CKKS_SLOT*GPQHE_CKKS_SLOT];
  COMPLEX U1_ConjTrans[GPQHE_CKKS_SLOT*GPQHE_CKKS_SLOT];
  COMPLEX U1_Trans[GPQHE_CKKS_SLOT*GPQHE_CKKS_SLOT];
  for (size_t i=0; i<GPQHE_CKKS_SLOT; i++) {
    for (size_t j=0; j<GPQHE_CKKS_SLOT; j++) {
      U0_ConjTrans[i*GPQHE_CKKS_SLOT+j] = ctx.CRT_UConjTrans[i*GPQHE_CKKS_SLOT+j];
      U1_ConjTrans[i*GPQHE_CKKS_SLOT+j] = ctx.CRT_UConjTrans[(i+GPQHE_CKKS_SLOT)*GPQHE_CKKS_SLOT+j];
      U0_Trans[i*GPQHE_CKKS_SLOT+j] = ctx.CRT_UTrans[i*GPQHE_CKKS_SLOT+j];
      U1_Trans[i*GPQHE_CKKS_SLOT+j] = ctx.CRT_UTrans[(i+GPQHE_CKKS_SLOT)*GPQHE_CKKS_SLOT+j];
    }
  }
#endif
  /* alloc */
  he_ct_t ct_conj, ct0, ct1;
  he_alloc_ct(&ct_conj);
  he_alloc_ct(&ct0);
  he_alloc_ct(&ct1);
  he_pt_t pt;
  he_alloc_pt(&pt);

  /* main */
  he_const_pt(&pt, 1/polyctx.n);
  he_copy_ct(&ct_conj, ct);
  he_conj(&ct_conj, ck);

  /* ct_real */
  he_gemv(&ct0, bootstrapctx.U0_conjT,  ct     , rk);
  he_gemv(&ct1, bootstrapctx.U0_T    , &ct_conj, rk);
  he_add(ct_real, &ct0, &ct1);
  he_mulpt(ct_real, ct_real, &pt);
  he_rs(ct_real);

  /* ct_imag */
  he_gemv(&ct0, bootstrapctx.U1_conjT,  ct     , rk);
  he_gemv(&ct1, bootstrapctx.U1_T    , &ct_conj, rk);
  he_add(ct_imag, &ct0, &ct1);
  he_mulpt(ct_imag, ct_imag, &pt);
  he_rs(ct_imag);

  /* release */
  he_free_ct(&ct_conj);
  he_free_ct(&ct0);
  he_free_ct(&ct1);
  he_free_pt(&pt);
}

/** slots to coefficients
 * Takes encryptions of (t_0, t_1, ..., t_(n/2-1)) and (t_(n/2), ..., t_(n-1))
 * before these vectors are encoded and transofmrs to an encryption of
 * t(x) = t_0 + t_1x + ... */
void he_slot2coeff(he_ct_t *ct,
                   const he_ct_t *ct0, const he_ct_t *ct1,
                   const he_evk_t *rk)
{
#if 0
  COMPLEX U0[GPQHE_CKKS_SLOT*GPQHE_CKKS_SLOT];
  COMPLEX U1[GPQHE_CKKS_SLOT*GPQHE_CKKS_SLOT];
  for (size_t i=0; i<GPQHE_CKKS_SLOT; i++) {
    for (size_t j=0; j<GPQHE_CKKS_SLOT; j++) {
      U0[i*GPQHE_CKKS_SLOT+j] = ctx.CRT_U[i*GPQHE_N+j];
      U1[i*GPQHE_CKKS_SLOT+j] = ctx.CRT_U[i*GPQHE_N+j+GPQHE_CKKS_SLOT];
    }
  }
#endif
  he_ct_t Uct0, Uct1;
  he_alloc_ct(&Uct0);
  he_alloc_ct(&Uct1);
  he_gemv(&Uct0, bootstrapctx.U0, ct0, rk);
  he_gemv(&Uct1, bootstrapctx.U1, ct1, rk);
  he_add(ct, &Uct0, &Uct1);
  /* release */
  he_free_ct(&Uct0);
  he_free_ct(&Uct1);
}

/**
 * he_ckks_rlsin - The relineared version of ckks_sin
 * 
 * Calculate sin(a*ct)/a = (exp(a*i*ct)-exp(-a*i*ct))/(2*i*a) 
 */ 
void he_rlsin(he_ct_t *ct_dest,
              const double a, const he_ct_t *ct,
              const he_evk_t *rlk, const he_evk_t *ck, const unsigned int iter)
{
  unsigned int slots = hectx.slots;
  _Complex double coeff[slots];
  /* alloc */
  he_pt_t pt;
  he_alloc_pt(&pt);
  he_ct_t ct_exp, ct_exp_neg;
  he_alloc_ct(&ct_exp);
  he_alloc_ct(&ct_exp_neg);

  /* exp(i*a*ct) */
  he_exp(&ct_exp, a*I, ct, rlk, iter);
  /* exp(-a*i*ct) */
  he_copy_ct(&ct_exp_neg, &ct_exp);
  he_conj(&ct_exp_neg, ck);
  /* exp(a*i*ct)-exp(-a*i*ct) == 2*i*sin(a*ct) */
  he_sub(ct_dest, &ct_exp, &ct_exp_neg);
  /* 1/(2*i*a) */
  _Complex double b = 1/(2*I*a);
  for (size_t i=0; i<slots; i++)
    coeff[i] = b;
  he_ecd(&pt, coeff);
  /* sin(a*ct)/a */
  he_mulpt(ct_dest, ct_dest, &pt);
  he_rs(ct_dest);
  he_free_ct(&ct_exp);
  he_free_ct(&ct_exp_neg);
  he_free_pt(&pt);
}

void he_bootstrap(he_ct_t *ct,
                  const he_evk_t *rlk, const he_evk_t *ck, const he_evk_t *rk,
                  const unsigned int iter)
{
  /**
   * +-----------+       +-----------+
   * |ct, ct_conj|======>| ct0 | ct1 |
   * +-----------+       +-----------+
   *      ^ |                |    |
   *      | V                V    V
   * +-----------+         +--+  +--+
   * |     pt    |         |pt|  |pt|
   * +-----------+         +--+  +--+
   *      ^ |                |    |
   *      | |                V    V
   *      | |            +-------------+
   *      | V            | dcdr | dcdi |
   * +-----------+       +-------------+
   * |    msg    |       | msgr | msgi |
   * +-----------+       +-------------+
   */
  /**
   * First of all, based on the well-known fact that UINT64_MAX=1.8446744e+19,
   * while DBL_MAX=1.7976931e+308, so a cascade product of 15 u64 numbers can
   * be represented by double, whereas 16 of them would cause `inf`.
   * 
   * Since the current modulus ql is the product of l limbs which do not excess
   * machine word size (u64), so l must smaller than 16, otherwise ql would be
   * represented as `inf` in double.
   */

  //double ql;
  /* Raise scaling factor. */
  double old_Delta = hectx.Delta;
  //unsigned int old_l = ct->l;
  
  /* raise ciphertext modulus from (mod ql) to (mod qL) */
  hectx.Delta = mpi_to_double(hectx.q[hectx.L]);
  ct->l  = hectx.L;
  ct->nu = hectx.Delta;

  /* alloc */
  he_ct_t ct0, ct1;
  he_alloc_ct(&ct0);
  he_alloc_ct(&ct1);

  /* Coefficient to Slot */
  he_show_ct_params(ct, "before coeff_to_slot");
  he_coeff2slot(&ct0, &ct1, ct, ck, rk);

  he_show_ct_params(&ct0, "ct0 after coeff_to_slot, before rlsin");
  he_show_ct_params(&ct1, "ct1 after coeff_to_slot, before rlsin");

  /* Exponentiate */
  double a = 2*GPQHE_PI;
  he_rlsin(&ct0, a, &ct0, rlk, ck, iter);
  he_rlsin(&ct1, a, &ct1, rlk, ck, iter);
  he_show_ct_params(&ct0, "ct0 after rlsin, before slot_to_coeff");
  he_show_ct_params(&ct1, "ct1 after rlsin, before slot_to_coeff");

  /* Slot to Coefficient */
  he_slot2coeff(ct, &ct0, &ct1, rk);
  he_show_ct_params(ct, "after slot_to_coeff");
  he_rs(ct);
  he_show_ct_params(ct, "finally rescale");

  /* Reset scaling factor */
  hectx.Delta = old_Delta;
  ct->nu = hectx.Delta;
}

#endif

END_DECLS
