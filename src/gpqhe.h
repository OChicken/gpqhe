/*
 * gpqhe.h - GNU post-quantum homomorphic-encryption library interface.
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

#ifndef GPQHE_H
#define GPQHE_H

#include "params.h"
#include "poly.h"

BEGIN_DECLS

struct bnd_ctx {
  double Bclean;    /* Freshly encrypted noise upper bound */
  double Brs;       /* rescale noise upper bound */
  double Bks;       /* key switch noise upper bound */
  double *Bmult;
};

struct he_ctx {
  MPI P;
  MPI PqL;
  MPI *q;           /* initial modulus, used for generating pk,sk,rlk,rk,ck. */
  MPI *qh;          /* q half */
  MPI p;
  unsigned int slots;
  double Delta;   /* scaling factor */
  unsigned int L;   /* maximum level */
  unsigned int dim; /* current number of limbs; current dimension of the RNS */
  unsigned int dimevk;
  /* bounds */
  struct bnd_ctx bnd;
  double Bmult;     /* valid multiplication */
};

struct bootstrap_ctx {
  //uint64_t **rp;
  double Delta;
  poly_rns_t *rp;
  poly_rns_t *rpinv;
  poly_rns_t rp1;
  poly_rns_t rp2;
  unsigned int *bnd;
  unsigned int *bndinv;
  unsigned int bnd1;
  unsigned int bnd2;
  unsigned int logp;
  _Complex double *U0;
  _Complex double *U1;
  _Complex double *U0_T;
  _Complex double *U1_T;
  _Complex double *U0_conjT;
  _Complex double *U1_conjT;
};

struct he_pk {
  poly_mpi_t p0;
  poly_mpi_t p1;
};
typedef struct he_pk he_pk_t;

struct he_evk {
  poly_rns_t p0;
  poly_rns_t p1;
};
typedef struct he_evk he_evk_t;

struct he_ct{
  unsigned int l;      /* Current evaluation level, 0<=l<=L */
  double nu;
  double B;
  poly_mpi_t c0;
  poly_mpi_t c1;
};
typedef struct he_ct he_ct_t;

struct he_pt{
  double nu;      /* canonical embedding norm */
  poly_mpi_t m;
};
typedef struct he_pt he_pt_t;

/* precomp.c */
void hectx_init(unsigned int logn, MPI q, unsigned int slots, uint64_t Delta);
void hectx_exit();

/* he-mem.c */
void he_alloc_sk(poly_mpi_t *sk);
void he_alloc_pk(struct he_pk *pk);
void he_alloc_evk(he_evk_t *evk);
void he_alloc_pt(struct he_pt *pt);
void he_alloc_ct(struct he_ct *ct);
void he_free_sk(poly_mpi_t *sk);
void he_free_pk(struct he_pk *pk);
void he_free_evk(he_evk_t *evk);
void he_free_pt(struct he_pt *pt);
void he_free_ct(struct he_ct *ct);
void he_copy_ct(struct he_ct *dest, const struct he_ct *src);

/* he-info.c */
void he_show_ctx_params();
void he_show_pt_params(const struct he_pt *pt, const char *title, ...);
void he_show_ct_params(const struct he_ct *ct, const char *title, ...);

/* he-encode.c */
void he_ecd(struct he_pt *pt, const _Complex double *m);
void he_dcd(_Complex double *m, struct he_pt *pt);
void he_const_pt(he_pt_t *pt, const _Complex double num);

/* he-kem.c */
void he_keypair(struct he_pk *pk, poly_mpi_t *sk);
void he_enc_pk(struct he_ct *ct, const struct he_pt *pt, const struct he_pk *pk);
void he_enc_sk(struct he_ct *ct, const struct he_pt *pt, const poly_mpi_t *sk);
void he_dec(struct he_pt *pt, const struct he_ct *ct, const poly_mpi_t *sk);
void he_genrlk(he_evk_t *rlk, const poly_mpi_t *sk);
void he_genck(he_evk_t *ck, const poly_mpi_t *sk);
void he_genrk(he_evk_t *rk, const poly_mpi_t *sk);

/* he-rescale.c */
void he_rs(struct he_ct *ct);
void he_moddown(he_ct_t *ct);

/* he-add.c */
void he_add(he_ct_t *ct, const he_ct_t *ct1, const he_ct_t *ct2);
void he_sub(he_ct_t *ct, const he_ct_t *ct1, const he_ct_t *ct2);
void he_addpt(he_ct_t *dest, const he_ct_t *src, const he_pt_t *pt);
void he_subpt(he_ct_t *dest, const he_ct_t *src, const he_pt_t *pt);
void he_neg(he_ct_t *ct);

/* he-mult.c */
void he_mul(he_ct_t *ct, const he_ct_t *ct1, const he_ct_t *ct2, const he_evk_t *rlk);
void he_mulpt(struct he_ct *dest, const struct he_ct *src, const struct he_pt *pt);

/* he-automorphism.c */
void he_conj(he_ct_t *ct, const he_evk_t *ck);
void he_rot(he_ct_t *ct, const int rot, const he_evk_t *rk);

/* he-algo.c */
void he_gemv(he_ct_t *ct_dest, const _Complex double *A, const he_ct_t *ct, const he_evk_t *rk);
void he_sum(he_ct_t *ct_sum, const he_ct_t *ct, const he_evk_t *rk);
void he_idx(he_ct_t *ct_sum, const he_ct_t *ct, const unsigned int idx, const he_evk_t *rk);
void he_nrm2(he_ct_t *ct_nrm2, const he_ct_t *ct, const he_evk_t *rlk, const he_evk_t *ck, const he_evk_t *rk);
void he_inv(he_ct_t *ct_inv, const he_ct_t *ct, const he_evk_t *rlk, const unsigned int iter);
void he_exp(he_ct_t *ct_exp, _Complex double a, const he_ct_t *ct, const he_evk_t *rlk, const unsigned int iter);
void he_sqrt(he_ct_t *ct_sqrt, const he_ct_t *ct, const he_evk_t *rlk, const unsigned int iter);
void he_sigmoid(he_ct_t *ct_sigmoid, const he_ct_t *ct, const he_evk_t *rlk);
void he_log(he_ct_t *ct_log, const he_ct_t *ct, const he_evk_t *rlk);
void he_cmp(he_ct_t *ct_cmp, const he_ct_t *ct1, const he_ct_t *ct2, const he_evk_t *rlk, const unsigned int iter, const unsigned int alpha);
void he_cmppt(he_ct_t *ct_cmp, const he_ct_t *ct, const he_pt_t *pt, const he_evk_t *rlk, const unsigned int iter, const unsigned int alpha);

/* he-bootstrap.c */
void he_bootstrapctx_init();
void he_bootstrapctx_exit();
void he_coeff2slot(he_ct_t *ct_real, he_ct_t *ct_imag,
                   const he_ct_t *ct,
                   const he_evk_t *ck, const he_evk_t *rk);
void he_rlsin(he_ct_t *ct_dest,
              const double a, const he_ct_t *ct,
              const he_evk_t *rlk, const he_evk_t *ck, const unsigned int iter);

END_DECLS

#endif /* GPQHE_H */
