/*
 * FHE operation collections of multiplication and rescale.
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

/** Relinearizes a dim3 ciphertext back down to dim2 */
static void he_relin(he_ct_t *ct,
  const poly_mpi_t *d0, const poly_mpi_t *d1, const poly_mpi_t *d2,
  const he_evk_t *rlk, const MPI ql)
{
  /* local variables */
  MPI qlh = mpi_copy(hectx.qh[ct->l]);
  MPI P   = mpi_copy(hectx.P);
  MPI Pql = mpi_new(0);
  mpi_mul(Pql, P, ql);
  unsigned int n = polyctx.n;
  /* d2 in Rql, rlk.{p0,p1} in R_PqL */
  unsigned int dim = (mpi_get_nbits(ql)+mpi_get_nbits(hectx.PqL)+polyctx.logn)/GPQHE_LOGP+1;
  /* decompose d2 */
  poly_rns_t d2hat, c0hat, c1hat;
  poly_rns_alloc(&d2hat, 1);
  poly_rns_alloc(&c0hat, dim);
  poly_rns_alloc(&c1hat, dim);
  struct rns_ctx *rns = polyctx.rns;
  for (unsigned int d=0; d<dim; d++) {
    rns_decompose(d2hat.coeffs, d2->coeffs, rns);
    ntt(d2hat.coeffs, rns);
    poly_rns_mul(&c0hat.coeffs[d*n], d2hat.coeffs, &rlk->p0.coeffs[d*n], rns);
    invntt(&c0hat.coeffs[d*n], rns); /* d2*rlk.p0, in R_{ql*PqL} */
    poly_rns_mul(&c1hat.coeffs[d*n], d2hat.coeffs, &rlk->p1.coeffs[d*n], rns);
    invntt(&c1hat.coeffs[d*n], rns); /* d2*rlk.p0, in R_{ql*PqL} */
    rns = (d<dim-1)? rns->next : rns;
  }
  poly_rns2mpi(&ct->c0, &c0hat, rns, Pql);
  poly_rns2mpi(&ct->c1, &c1hat, rns, Pql);
  /* main */
  for (size_t i=0; i<n; i++) {
    mpi_rdiv(ct->c0.coeffs[i], ct->c0.coeffs[i], P); /* (d2*rlk.p0)/P, now in Rql */
    mpi_rdiv(ct->c1.coeffs[i], ct->c1.coeffs[i], P); /* (d2*rlk.p1)/P, now in Rql */
    mpi_addm(ct->c0.coeffs[i], ct->c0.coeffs[i], d0->coeffs[i], ql);
    mpi_addm(ct->c1.coeffs[i], ct->c1.coeffs[i], d1->coeffs[i], ql);
    mpi_smod(ct->c0.coeffs[i], ql, qlh);
    mpi_smod(ct->c1.coeffs[i], ql, qlh);
  }
  /* release */
  mpi_release(qlh);
  mpi_release(P);
  mpi_release(Pql);
  poly_rns_free(&d2hat);
  poly_rns_free(&c0hat);
  poly_rns_free(&c1hat);
}

/** ct_dest = ct1*ct2 */
void he_mul(he_ct_t *ct, const he_ct_t *ct1, const he_ct_t *ct2, const he_evk_t *rlk)
{
  assert(ct1->l==ct2->l);
  /* ct init */
  ct->l  = ct1->l;
  ct->nu = ct1->nu*ct2->nu;
  ct->B  = (ct1->nu)*(ct2->B) + (ct2->nu)*(ct1->B) + (ct1->B)*(ct2->B)
         + hectx.bnd.Bmult[ct->l];
  /* local variables */
  MPI q  = mpi_copy(hectx.q [ct->l]);
  unsigned int n = polyctx.n;
  unsigned int dim = (mpi_get_nbits(q)*2+polyctx.logn)/GPQHE_LOGP+1;
  /* alloc */
  poly_mpi_t d0, d1, d2;
  poly_mpi_alloc(&d0);
  poly_mpi_alloc(&d1);
  poly_mpi_alloc(&d2);
  poly_rns_t ct1c0hat, ct1c1hat, ct2c0hat, ct2c1hat;
  poly_rns_alloc(&ct1c0hat, 1);
  poly_rns_alloc(&ct1c1hat, 1);
  poly_rns_alloc(&ct2c0hat, 1);
  poly_rns_alloc(&ct2c1hat, 1);
  poly_rns_t d0hat, d1hat, d2hat;
  poly_rns_alloc(&d0hat, dim);
  poly_rns_alloc(&d1hat, dim);
  poly_rns_alloc(&d2hat, dim);
  /* main - cross terms */
  struct rns_ctx *rns = polyctx.rns;
  for (unsigned int d=0; d<dim; d++) {
    rns_decompose(ct1c0hat.coeffs, ct1->c0.coeffs, rns);
    rns_decompose(ct1c1hat.coeffs, ct1->c1.coeffs, rns);
    rns_decompose(ct2c0hat.coeffs, ct2->c0.coeffs, rns);
    rns_decompose(ct2c1hat.coeffs, ct2->c1.coeffs, rns);
    ntt(ct1c0hat.coeffs, rns);
    ntt(ct1c1hat.coeffs, rns);
    ntt(ct2c0hat.coeffs, rns);
    ntt(ct2c1hat.coeffs, rns);
    poly_rns_mul( &d0hat.coeffs[d*n], ct1c0hat.coeffs, ct2c0hat.coeffs, rns);
    invntt(&d0hat.coeffs[d*n], rns); /* d0 = ct1.c0*ct2.c0 */
    poly_rns_mul( &d2hat.coeffs[d*n], ct1c1hat.coeffs, ct2c1hat.coeffs, rns);
    invntt(&d2hat.coeffs[d*n], rns); /* d2 = ct1.c1*ct2.c1 */
    /* ct1.c0*ct2.c1 */
    poly_rns_mul(ct1c0hat.coeffs, ct1c0hat.coeffs, ct2c1hat.coeffs, rns);
    invntt(ct1c0hat.coeffs, rns);
    /* ct1.c1*ct2.c0 */
    poly_rns_mul(ct1c1hat.coeffs, ct1c1hat.coeffs, ct2c0hat.coeffs, rns);
    invntt(ct1c1hat.coeffs, rns);
    /* d1 = ct1.c0*ct2.c1 + ct1.c1*ct2.c0 */
    poly_rns_add(&d1hat.coeffs[d*n], ct1c0hat.coeffs, ct1c1hat.coeffs, rns);
    rns = (d<dim-1)? rns->next : rns;
  }
  poly_rns2mpi(&d0, &d0hat, rns, q); /* d0 = ct1.c0*ct2.c0 */
  poly_rns2mpi(&d2, &d2hat, rns, q); /* d2 = ct1.c1*ct2.c1 */
  poly_rns2mpi(&d1, &d1hat, rns, q); /* d1 = ct1.c0*ct2.c1 + ct1.c1*ct2.c0 */
  /* main - relinearize */
  he_relin(ct, &d0, &d1, &d2, rlk, q);
  /* release */
  mpi_release(q);
  poly_rns_free(&ct1c0hat);
  poly_rns_free(&ct1c1hat);
  poly_rns_free(&ct2c0hat);
  poly_rns_free(&ct2c1hat);
  poly_rns_free(&d0hat);
  poly_rns_free(&d1hat);
  poly_rns_free(&d2hat);
  poly_mpi_free(&d0);
  poly_mpi_free(&d1);
  poly_mpi_free(&d2);
}

/** ct_dest = ct*pt */
void he_mulpt(he_ct_t *dest, const he_ct_t *src, const he_pt_t *pt)
{
  /* ct init */
  dest->l  = src->l;
  dest->nu = src->nu*pt->nu;
  dest->B  = src->B*pt->nu;
  /* local varables */
  MPI q  = mpi_copy(hectx.q [dest->l]);
  unsigned int n = polyctx.n;
  unsigned int dim = (mpi_get_nbits(q)+log2(pt->nu)+polyctx.logn)/GPQHE_LOGP+1;
  /* alloc */
  poly_rns_t pthat, c0hat, c1hat;
  poly_rns_alloc(&pthat, 1);
  poly_rns_alloc(&c0hat, dim);
  poly_rns_alloc(&c1hat, dim);
  /* main */
  struct rns_ctx *rns = polyctx.rns;
  for (unsigned int d=0; d<dim; d++) {
    rns_decompose(pthat.coeffs, pt->m.coeffs, rns);
    rns_decompose(&c0hat.coeffs[d*n], src->c0.coeffs, rns);
    rns_decompose(&c1hat.coeffs[d*n], src->c1.coeffs, rns);
    ntt(pthat.coeffs, rns);
    ntt(&c0hat.coeffs[d*n], rns);
    ntt(&c1hat.coeffs[d*n], rns);
    poly_rns_mul(&c0hat.coeffs[d*n], &c0hat.coeffs[d*n], pthat.coeffs, rns);
    invntt(&c0hat.coeffs[d*n], rns);
    poly_rns_mul(&c1hat.coeffs[d*n], &c1hat.coeffs[d*n], pthat.coeffs, rns);
    invntt(&c1hat.coeffs[d*n], rns);
    rns = (d<dim-1)? rns->next : rns;
  }
  poly_rns2mpi(&dest->c0, &c0hat, rns, q);
  poly_rns2mpi(&dest->c1, &c1hat, rns, q);
  /* release */
  mpi_release(q);
  poly_rns_free(&pthat);
  poly_rns_free(&c0hat);
  poly_rns_free(&c1hat);
}

END_DECLS
