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

/** rescale */
void he_rs(struct he_ct *ct)
{
  /* ct init */
  ct->l -= 1;
  ct->nu /= hectx.Delta;
  ct->B = ct->B/hectx.Delta + hectx.bnd.Brs;
  /* local varables */
  unsigned int n = polyctx.n;
  MPI q  = mpi_copy(hectx.q [ct->l]);
  MPI qh = mpi_copy(hectx.qh[ct->l]);
  MPI p  = mpi_copy(hectx.p);
  /* main */
  for (unsigned int i=0; i<n; i++) {
    mpi_rdiv(ct->c0.coeffs[i], ct->c0.coeffs[i], p);
    mpi_rdiv(ct->c1.coeffs[i], ct->c1.coeffs[i], p);
    mpi_smod(ct->c0.coeffs[i], q, qh);
    mpi_smod(ct->c1.coeffs[i], q, qh);
  }
  mpi_release(p);
  mpi_release(q);
  mpi_release(qh);
}

void he_moddown(he_ct_t *ct)
{
  /* ct init */
  ct->l -= 1;
  /* local varables */
  unsigned int n = polyctx.n;
  MPI q = mpi_copy(hectx.q [ct->l]);
  MPI qh = mpi_copy(hectx.qh[ct->l]);
  for (unsigned int i=0; i<n; i++) {
    mpi_smod(ct->c0.coeffs[i], q, qh);
    mpi_smod(ct->c1.coeffs[i], q, qh);
  }
  mpi_release(q);
  mpi_release(qh);
}

END_DECLS
