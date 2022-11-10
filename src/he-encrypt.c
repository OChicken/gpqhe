/*
 * Encryptor.
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

BEGIN_DECLS

/* polyctx.h */
extern struct poly_ctx polyctx;

/* fhe.h */
extern struct he_ctx hectx;

/* sample.c */
extern void sample_error(poly_mpi_t *r);
extern void sample_zo(poly_mpi_t *r);
extern void sample_uniform(poly_mpi_t *r, const MPI q);

void he_enc_pk(struct he_ct *ct, const struct he_pt *pt, const struct he_pk *pk)
{
  /* ct init */
  ct->l  = hectx.L;
  ct->nu = (pt->nu>=hectx.Delta)? pt->nu : hectx.Delta;
  ct->B  = hectx.bnd.Bclean;
  /* local variables */
  MPI q  = mpi_copy(hectx.q[hectx.L]);
  MPI qh = mpi_copy(hectx.qh[hectx.L]);
  unsigned int n = polyctx.n;
  /* sample coeffs for pk */
  poly_mpi_t v;
  poly_mpi_alloc(&v);
  sample_zo(&v);
  /* sample error for pt */
  poly_mpi_t e0, e1;
  poly_mpi_alloc(&e0);
  poly_mpi_alloc(&e1);
  sample_error(&e0);
  sample_error(&e1);
  /* main */
  poly_mul(&ct->c0, &pk->p0, &v, hectx.dim, q);
  poly_mul(&ct->c1, &pk->p1, &v, hectx.dim, q);
  for (unsigned int i=0; i<n; i++) {
    mpi_add(ct->c0.coeffs[i], ct->c0.coeffs[i], pt->m.coeffs[i]);
    mpi_addm(ct->c0.coeffs[i], ct->c0.coeffs[i], e0.coeffs[i], q);
    mpi_addm(ct->c1.coeffs[i], ct->c1.coeffs[i], e1.coeffs[i], q);
    mpi_smod(ct->c0.coeffs[i], q, qh); /* (v*pk + pt + e0) smod q */
    mpi_smod(ct->c1.coeffs[i], q, qh); /* (v*pk      + e1) smod q */
  }
  /* release */
  mpi_release(q);
  mpi_release(qh);
  poly_mpi_free(&v);
  poly_mpi_free(&e0);
  poly_mpi_free(&e1);
}

void he_enc_sk(struct he_ct *ct, const struct he_pt *pt, const poly_mpi_t *sk)
{
  /* ct init */
  ct->l  = hectx.L;
  ct->nu = (pt->nu>=hectx.Delta)? pt->nu : hectx.Delta;
  ct->B  = hectx.bnd.Bclean;
  /* local variables */
  MPI q  = mpi_copy(hectx.q[hectx.L]);
  MPI qh = mpi_copy(hectx.qh[hectx.L]);
  unsigned int n = polyctx.n;
  /* sample */
  poly_mpi_t e;
  poly_mpi_alloc(&e);
  sample_error(&e);
  sample_uniform(&ct->c1, q);
  /* main */
  poly_mul(&ct->c0, &ct->c1, sk, hectx.dim, q); /* c0=c1*sk */
  for (unsigned int i=0; i<n; i++) {
    mpi_neg (ct->c0.coeffs[i], ct->c0.coeffs[i]); /* c0=-c1*sk */
    mpi_add (ct->c0.coeffs[i], ct->c0.coeffs[i], pt->m.coeffs[i]); /* c0=-c1*sk+m */
    mpi_addm(ct->c0.coeffs[i], ct->c0.coeffs[i], e.coeffs[i], q); /* (-c1*sk+m+e) smod q */
    mpi_smod(ct->c0.coeffs[i], q, qh);
    mpi_smod(ct->c1.coeffs[i], q, qh);
  }
  /* release */
  mpi_release(q);
  mpi_release(qh);
  poly_mpi_free(&e);
}

void he_dec(struct he_pt *pt, const struct he_ct *ct, const poly_mpi_t *sk)
{
  /* pt init */
  pt->nu = ct->nu;
  /* local variables */
  MPI q  = mpi_copy(hectx.q[ct->l]);
  MPI qh = mpi_copy(hectx.qh[ct->l]);
  unsigned int n = polyctx.n;
  unsigned int dim = mpi_get_nbits(q)/GPQHE_LOGP+1;
  /* main */
  poly_mul(&pt->m, &ct->c1, sk, dim, q); /* m=c1*sk */
  for (unsigned int i=0; i<n; i++) {
    mpi_addm(pt->m.coeffs[i], pt->m.coeffs[i], ct->c0.coeffs[i], q); /* m=c1*sk+c0 */
    mpi_smod(pt->m.coeffs[i], q, qh); /* m = c1*sk+c0 smod q */
  }
  /* release */
  mpi_release(q);
  mpi_release(qh);
}

END_DECLS
