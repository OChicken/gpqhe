/*
 * FHE memory management utils.
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

void he_alloc_sk(poly_mpi_t *sk)
{
  poly_mpi_alloc(sk);
}

void he_alloc_pk(struct he_pk *pk)
{
  poly_mpi_alloc(&pk->p0);
  poly_mpi_alloc(&pk->p1);
}

void he_alloc_evk(he_evk_t *evk)
{
  poly_rns_alloc(&evk->p0, hectx.dimevk);
  poly_rns_alloc(&evk->p1, hectx.dimevk);
}

void he_alloc_pt(struct he_pt *pt)
{
  poly_mpi_alloc(&pt->m);
}

void he_alloc_ct(struct he_ct *ct)
{
  poly_mpi_alloc(&ct->c0);
  poly_mpi_alloc(&ct->c1);
}

void he_free_sk(poly_mpi_t *sk)
{
  poly_mpi_free(sk);
}

void he_free_pk(struct he_pk *pk)
{
  poly_mpi_free(&pk->p0);
  poly_mpi_free(&pk->p1);
}

void he_free_evk(he_evk_t *evk)
{
  poly_rns_free(&evk->p0);
  poly_rns_free(&evk->p1);
}

void he_free_pt(struct he_pt *pt)
{
  poly_mpi_free(&pt->m);
}

void he_free_ct(struct he_ct *ct)
{
  poly_mpi_free(&ct->c0);
  poly_mpi_free(&ct->c1);
}

/** copy ciphertext */
void he_copy_ct(struct he_ct *dest, const struct he_ct *src)
{
  dest->l  = src->l;
  dest->nu = src->nu;
  dest->B  = src->B;
  for (size_t i=0; i<polyctx.n; i++) {
    mpi_set(dest->c0.coeffs[i], src->c0.coeffs[i]);
    mpi_set(dest->c1.coeffs[i], src->c1.coeffs[i]);
  }
}

END_DECLS
