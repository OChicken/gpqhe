/*
 * FHE operation collections of addition and subtraction.
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

/** ct = ct1+ct2 */
void he_add(he_ct_t *ct, const he_ct_t *ct1, const he_ct_t *ct2)
{
  /* ct init */
  assert(ct1->l==ct2->l);
  ct->l  = ct1->l;
  ct->nu = (ct1->nu>=ct2->nu)? ct1->nu : ct2->nu;//ct1->nu+ct2->nu;
  ct->B  = ct1->B+ct2->B ;
  /* local variables */
  MPI q  = mpi_copy(hectx.q [ct->l]);
  MPI qh = mpi_copy(hectx.qh[ct->l]);
  unsigned int n = polyctx.n;
  /* main */
  for (unsigned int i=0; i<n; i++) {
    mpi_addm(ct->c0.coeffs[i], ct1->c0.coeffs[i], ct2->c0.coeffs[i], q);
    mpi_addm(ct->c1.coeffs[i], ct1->c1.coeffs[i], ct2->c1.coeffs[i], q);
    mpi_smod(ct->c0.coeffs[i], q, qh);
    mpi_smod(ct->c1.coeffs[i], q, qh);
  }
  /* release */
  mpi_release(q);
  mpi_release(qh);
}

/** ct = ct1-ct2 */
void he_sub(he_ct_t *ct, const he_ct_t *ct1, const he_ct_t *ct2)
{
  /* ct init */
  assert(ct1->l==ct2->l);
  ct->l  = ct1->l;
  ct->nu = (ct1->nu>=ct2->nu)? ct1->nu : ct2->nu;//ct1->nu+ct2->nu;
  ct->B  = ct1->B+ct2->B ;
  /* local variables */
  MPI q  = mpi_copy(hectx.q [ct->l]);
  MPI qh = mpi_copy(hectx.qh[ct->l]);
  unsigned int n = polyctx.n;
  /* main */
  for (unsigned int i=0; i<n; i++) {
    mpi_subm(ct->c0.coeffs[i], ct1->c0.coeffs[i], ct2->c0.coeffs[i], q);
    mpi_subm(ct->c1.coeffs[i], ct1->c1.coeffs[i], ct2->c1.coeffs[i], q);
    mpi_smod(ct->c0.coeffs[i], q, qh);
    mpi_smod(ct->c1.coeffs[i], q, qh);
  }
  /* release */
  mpi_release(q);
  mpi_release(qh);
}

/** dest = src+pt */
void he_addpt(he_ct_t *dest, const he_ct_t *src, const he_pt_t *pt)
{
  /* ct init */
  dest->l  = src->l;
  dest->nu = (src->nu>=pt->nu)? src->nu : pt->nu;//src->nu+pt->nu;
  dest->B  = src->B;
  /* local variables */
  MPI q  = mpi_copy(hectx.q [dest->l]);
  MPI qh = mpi_copy(hectx.qh[dest->l]);
  unsigned int n = polyctx.n;
  /* main */
  for (unsigned int i=0; i<n; i++) {
    mpi_addm(dest->c0.coeffs[i], src->c0.coeffs[i], pt->m.coeffs[i], q);
    mpi_mod(dest->c1.coeffs[i], src->c1.coeffs[i], q);
    mpi_smod(dest->c0.coeffs[i], q, qh);
    mpi_smod(dest->c1.coeffs[i], q, qh);
  }
  /* release */
  mpi_release(q);
  mpi_release(qh);
}

/** dest = src-pt */
void he_subpt(he_ct_t *dest, const he_ct_t *src, const he_pt_t *pt)
{
  /* ct init */
  dest->l  = src->l;
  dest->nu = (src->nu>=pt->nu)? src->nu : pt->nu;//src->nu+pt->nu;
  dest->B  = src->B;
  /* local variables */
  MPI q  = mpi_copy(hectx.q [dest->l]);
  MPI qh = mpi_copy(hectx.qh[dest->l]);
  unsigned int n = polyctx.n;
  /* main */
  for (unsigned int i=0; i<n; i++) {
    mpi_subm(dest->c0.coeffs[i], src->c0.coeffs[i], pt->m.coeffs[i], q);
    mpi_mod(dest->c1.coeffs[i], src->c1.coeffs[i], q);
    mpi_smod(dest->c0.coeffs[i], q, qh);
    mpi_smod(dest->c1.coeffs[i], q, qh);
  }
  /* release */
  mpi_release(q);
  mpi_release(qh);
}

/** -ct */
void he_neg(he_ct_t *ct)
{
  /* local variables */
  MPI q  = mpi_copy(hectx.q [ct->l]);
  MPI qh = mpi_copy(hectx.qh[ct->l]);
  unsigned int n = polyctx.n;
  /* main */
  for (unsigned int i=0; i<n; i++) {
    mpi_neg(ct->c0.coeffs[i], ct->c0.coeffs[i]);
    mpi_neg(ct->c1.coeffs[i], ct->c1.coeffs[i]);
    mpi_smod(ct->c0.coeffs[i], q, qh);
    mpi_smod(ct->c1.coeffs[i], q, qh);
  }
  /* release */
  mpi_release(q);
  mpi_release(qh);
}

END_DECLS
