/*
 * kem - key encapsulation mechanism.
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

#include "gpqhe.h"

BEGIN_DECLS

/* poly.h */
extern struct poly_ctx polyctx;

/* fhe.h */
extern struct he_ctx hectx;

/* sample.c */
extern void sample_sk(poly_mpi_t *r);
extern void sample_error(poly_mpi_t *r);
extern void sample_uniform(poly_mpi_t *r, const MPI q);

/* ntt.c */
extern void ntt(uint64_t a[], const struct rns_ctx *rns);

/* rns.c */
extern void rns_decompose(uint64_t ahat[], const MPI a[], const struct rns_ctx *rns);

/** he_keypair - sk and pk key pair */
void he_keypair(he_pk_t *pk, poly_mpi_t *sk)
{
  /* local variables */
  MPI qL = mpi_copy(hectx.q[hectx.L]);
  MPI qh = mpi_copy(hectx.qh[hectx.L]);
  unsigned int n = polyctx.n;
  printf("Generating sk and pk ... ");
  fflush(stdout);
  /* secret key */
  sample_sk(sk);
  /* sample error for pk */
  poly_mpi_t e;
  poly_mpi_alloc(&e);
  sample_error(&e);
  /* main */
  sample_uniform(&pk->p1, qL);
  poly_mul(&pk->p0, sk, &pk->p1, hectx.dim, qL);
  for (unsigned int i=0; i<n; i++) {
    mpi_neg(pk->p0.coeffs[i], pk->p0.coeffs[i]);
    mpi_addm(pk->p0.coeffs[i], pk->p0.coeffs[i], e.coeffs[i], qL);
    mpi_smod(pk->p0.coeffs[i], qL, qh);
    mpi_smod(pk->p1.coeffs[i], qL, qh); /* add this or not does not affect the correctness */
  }
  /* release */
  mpi_release(qL);
  mpi_release(qh);
  poly_mpi_free(&e);
  printf("done.\n");
}

/** switching keys */
static void he_genswk(he_evk_t *swk, const poly_mpi_t *sp, const poly_mpi_t *sk)
{
  /* local variables */
  MPI P    = mpi_copy(hectx.P);
  MPI PqL  = mpi_copy(hectx.PqL);
  MPI PqLh = mpi_new(0);
  MPI two  = mpi_set_ui(NULL, 2);
  mpi_fdiv(PqLh, NULL, PqL, two);
  unsigned int n = polyctx.n;
  unsigned int dim = (mpi_get_nbits(PqL)+polyctx.logn)/GPQHE_LOGP+1;
  /* sample error for swk */
  poly_mpi_t e;
  poly_mpi_alloc(&e);
  sample_error(&e);
  /* main */
  for (size_t i=0; i<n; i++)
    mpi_mul(sp->coeffs[i], sp->coeffs[i], P);
  poly_mpi_t swkp0, swkp1;
  poly_mpi_alloc(&swkp0);
  poly_mpi_alloc(&swkp1);
  sample_uniform(&swkp1, PqL);
  poly_mul(&swkp0, &swkp1, sk, dim, PqL);
  for (unsigned int i=0; i<n; i++) {
    mpi_neg(swkp0.coeffs[i], swkp0.coeffs[i]);                      /* swk.p0 = -swk.p1*sk */
    mpi_add(swkp0.coeffs[i], swkp0.coeffs[i], e.coeffs[i]);         /* swk.p0 = -swk.p1*sk+e */
    mpi_addm(swkp0.coeffs[i], swkp0.coeffs[i], sp->coeffs[i], PqL); /* swk.p0 = -swk.p1*sk+e+Psp mod PqL */
    mpi_smod(swkp0.coeffs[i], PqL, PqLh);
    mpi_smod(swkp1.coeffs[i], PqL, PqLh);
  }
  struct rns_ctx *rns = polyctx.rns;
  for (unsigned int d=0; d<hectx.dimevk; d++) {
    rns_decompose(&swk->p0.coeffs[d*n], swkp0.coeffs, rns);
    ntt(&swk->p0.coeffs[d*n], rns);
    rns_decompose(&swk->p1.coeffs[d*n], swkp1.coeffs, rns);
    ntt(&swk->p1.coeffs[d*n], rns);
    rns = (d<hectx.dimevk-1)? rns->next : rns;
  }
  mpi_release(P);
  mpi_release(PqL);
  mpi_release(PqLh);
  mpi_release(two);
  poly_mpi_free(&e);
  poly_mpi_free(&swkp0);
  poly_mpi_free(&swkp1);
}

void he_genrlk(he_evk_t *rlk, const poly_mpi_t *sk)
{
  /* local variables */
  MPI q = mpi_copy(hectx.q[hectx.L]);
  poly_mpi_t s2;
  poly_mpi_alloc(&s2);
  /* main */
  printf("Generating rlk ... ");
  fflush(stdout);
  /* relinearization key */
  poly_mul(&s2, sk, sk, mpi_get_nbits(q)/GPQHE_LOGP+1, q);
  he_genswk(rlk, &s2, sk);
  printf("done.\n");
  /* release */
  mpi_release(q);
  poly_mpi_free(&s2);
}

/** conjugate key */
void he_genck(he_evk_t *ck, const poly_mpi_t *sk)
{
  /* local variables */
  poly_mpi_t ck_;
  poly_mpi_alloc(&ck_);
  /* conjugate key */
  printf("Generating ck ... ");
  fflush(stdout);
  poly_conj(&ck_, sk);
  he_genswk(ck, &ck_, sk);
  printf("done.\n");
  /* release */
  poly_mpi_free(&ck_);
}

void he_genrk(he_evk_t *rk, const poly_mpi_t *sk)
{
  /* local variables */
  poly_mpi_t rk_;
  poly_mpi_alloc(&rk_);
  /* rotation keys */
  printf("Generating rk ... ");
  fflush(stdout);
  for (unsigned int rot=0; rot<hectx.slots; rot++) {
    poly_rot(&rk_, sk, rot);
    he_genswk(&rk[rot], &rk_, sk);
  }
  printf("done.\n");
  /* release */
  poly_mpi_free(&rk_);
}

END_DECLS
