/*
 * FHE info fetching utils.
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

/* types.c */
extern void show_mpi(MPI a);

void he_show_ctx_params()
{
  printf("\033[1mEncryption parameters:\033[0m\n");
#if GPQHE_CQ == 'C'
  printf("\\_ Security type: classical %u bits.\n", GPQHE_SEC_LEVEL);
#elif GPQHE_CQ == 'Q'
  printf("\\_ Security type: quantum %u bits.\n", GPQHE_SEC_LEVEL);
#endif
  printf("\\_ Polynomial degree n = %u\n", polyctx.n);
  printf("\\_ Cyclotomic index m = %u\n", polyctx.m);
  //printf("\\_ Ring constant cM = %f\n", ctx.cM);

  /* Q = q0*q1*...*qL */
  printf("\\_ Ciphertext modulus ql = q0*p^l:\n");
  printf("   \\_ L = %u\n", hectx.L);
  printf("   \\_ q0 (%u bits) = %#02lx\n", mpi_get_nbits(hectx.q[0]), mpi_to_u64(hectx.q[0]));
  printf("   \\_ p  (%u bits) = %#02lx\n", mpi_get_nbits(hectx.p), mpi_to_u64(hectx.p));
  printf("   \\_ qL (%u bits) = ", mpi_get_nbits(hectx.q[hectx.L]));
  show_mpi(hectx.q[hectx.L]);

  /* P = p1*p2*...*pn */
  printf("\\_ RNS modulus P = p1...pn:\n");
  printf("   \\_ dim = %u\n", hectx.dim);
  printf("   \\_ p = { ");
  struct rns_ctx *rns=polyctx.rns;
  for (unsigned int d=0; d<hectx.dim; d++) {
    printf("%#02lx ", rns->p);
    rns = rns->next;
  }
  printf("}\n");
  printf("   \\_ P (%u bits) = ", mpi_get_nbits(hectx.P));
  show_mpi(hectx.P);

  /* Bounds */
  printf("\\_ Bounds:\n");
  struct bnd_ctx *bnd=&hectx.bnd;
  printf("   \\_ Bclean = %f (%u bits)\n", bnd->Bclean, (unsigned int)log2(bnd->Bclean)+1);
  printf("   \\_ Brs    = %f (%u bits)\n", bnd->Brs, (unsigned int)log2(bnd->Brs)+1);
  printf("   \\_ Bks    = %f (%u bits)\n", bnd->Bks, (unsigned int)log2(bnd->Bks)+1);
  printf("   \\_ Bmult  = { ");
  for (unsigned int l=0; l<=hectx.L; l++)
    printf("%f (%u bits) ", bnd->Bmult[l], (unsigned int)log2(bnd->Bmult[l])+1);
  printf("}\n");
  //printf("\\_ Number of primes for CRT context: %lu\n", ctx->num_p);
}

void he_show_pt_params(const struct he_pt *pt, const char *title, ...)
{
  va_list arg_ptr;
  printf("\033[1m\033[4m");
  va_start (arg_ptr, title) ;
  vfprintf (stdout, title, arg_ptr);
  va_end (arg_ptr);
  printf("\033[0m\n");
  printf("\\_ nu = %.10g (%ld bits)\n", pt->nu, (long)log2(pt->nu)+1);
}

void he_show_ct_params(const struct he_ct *ct, const char *title, ...)
{
  va_list arg_ptr;
  printf("\033[1m\033[4m");
  va_start (arg_ptr, title) ;
  vfprintf (stdout, title, arg_ptr);
  va_end (arg_ptr);
  printf("\033[0m\n");
  printf("\\_ l  = %u\n", ct->l);
  printf("\\_ nu = %.10g (%ld bits)\n", ct->nu, (long)log2(ct->nu)+1);
  printf("\\_ B  = %f\n", ct->B);
  printf("\\_ log(ql) = %u\n", mpi_get_nbits(hectx.q[ct->l])-1);
}

END_DECLS
