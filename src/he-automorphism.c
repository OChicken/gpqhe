/*
 * FHE operation collections of Galois automorphism: conjugate and rotate.
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
#include <math.h>

BEGIN_DECLS

/* poly.h */
extern struct poly_ctx polyctx;

/* fhe.h */
extern struct he_ctx hectx;

/* ntt.c */
extern void    ntt(uint64_t a[], const struct rns_ctx *rns);
extern void invntt(uint64_t a[], const struct rns_ctx *rns);

/* rns.c */
extern void rns_decompose(uint64_t ahat[], const MPI a[], const struct rns_ctx *rns);

/** Key-switching procedure */
static void he_swk(he_ct_t *ct,
  const poly_mpi_t *d0, const poly_mpi_t *d1,
  const he_evk_t *swk)
{
  /* local variables */
  MPI ql = mpi_copy(hectx.q[ct->l]);
  MPI qlh = mpi_copy(hectx.qh[ct->l]);
  MPI P   = mpi_copy(hectx.P);
  MPI Pql = mpi_new(0);
  mpi_mul(Pql, P, ql);
  unsigned int n = polyctx.n;
  /* d1 in Rql, swk.{p0,p1} in R_PqL */
  unsigned int dim = (mpi_get_nbits(ql)+mpi_get_nbits(hectx.PqL)+polyctx.logn)/GPQHE_LOGP+1;
  /* main */
  poly_rns_t d1hat, c0hat, c1hat;
  poly_rns_alloc(&d1hat, 1);
  poly_rns_alloc(&c0hat, dim);
  poly_rns_alloc(&c1hat, dim);
  struct rns_ctx *rns = polyctx.rns;
  for (unsigned int d=0; d<dim; d++) {
    rns_decompose(d1hat.coeffs, d1->coeffs, rns);
    ntt(d1hat.coeffs, rns);
    poly_rns_mul(&c0hat.coeffs[d*n], d1hat.coeffs, &swk->p0.coeffs[d*n], rns);
    invntt(&c0hat.coeffs[d*n], rns); /* d1*swk.p0, in R_{ql*PqL} */
    poly_rns_mul(&c1hat.coeffs[d*n], d1hat.coeffs, &swk->p1.coeffs[d*n], rns);
    invntt(&c1hat.coeffs[d*n], rns); /* d1*swk.p0, in R_{ql*PqL} */
    rns = (d<dim-1)? rns->next : rns;
  }
  poly_rns2mpi(&ct->c0, &c0hat, rns, Pql);
  poly_rns2mpi(&ct->c1, &c1hat, rns, Pql);
  for (unsigned int i=0; i<n; i++) {
    mpi_rdiv(ct->c0.coeffs[i], ct->c0.coeffs[i], P); /* (d1*swk.p0)/P, now in Rql */
    mpi_rdiv(ct->c1.coeffs[i], ct->c1.coeffs[i], P); /* (d1*swk.p1)/P, now in Rql */
    mpi_addm(ct->c0.coeffs[i], ct->c0.coeffs[i], d0->coeffs[i], ql);
    mpi_smod(ct->c0.coeffs[i], ql, qlh);
    mpi_smod(ct->c1.coeffs[i], ql, qlh);
  }
  /* release */
  mpi_release(ql);
  mpi_release(qlh);
  mpi_release(P);
  mpi_release(Pql);
  poly_rns_free(&d1hat);
  poly_rns_free(&c0hat);
  poly_rns_free(&c1hat);
}

void he_conj(he_ct_t *ct, const he_evk_t *ck)
{
  /* local varables */
  poly_mpi_t d0, d1;
  poly_mpi_alloc(&d0);
  poly_mpi_alloc(&d1);
  /* main */
  poly_conj(&d0, &ct->c0);
  poly_conj(&d1, &ct->c1);
  he_swk(ct, &d0, &d1, ck);
  /* release */
  poly_mpi_free(&d0);
  poly_mpi_free(&d1);
}

void he_rot(he_ct_t *ct, const int rot, const he_evk_t *rk)
{
  /* local variables */
  poly_mpi_t d0, d1;
  poly_mpi_alloc(&d0);
  poly_mpi_alloc(&d1);
  /* main */
  poly_rot(&d0, &ct->c0, rot);
  poly_rot(&d1, &ct->c1, rot);
  he_swk(ct, &d0, &d1, &rk[rot]);
  /* release */
  poly_mpi_free(&d0);
  poly_mpi_free(&d1);
}

END_DECLS
