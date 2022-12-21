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
#include <complex.h>
BEGIN_DECLS

/* fhe.h */
extern struct he_ctx hectx;

static void zrotdiag(_Complex double rotdiag[], const _Complex double A[],
                          const unsigned int idx, const int rot)
{
  const unsigned int m = hectx.slots;
  _Complex double diag[m];
  for (unsigned int i=0; i<m; i++)
    diag[i] = A[(i%m)*m + (idx+i)%m];
  for (unsigned int i=0; i<m; i++) {
    int rotidx = ((int)i+rot)%m;
    if (rotidx<0)
      rotidx += m;
    assert(rotidx>=0);
    rotdiag[i] = diag[rotidx];
  }
}

/** gemv: GEneral Matrix Vector multiplication
 * he_ckks_gemv(&ct_Av, A, &ct_v, evk->rot_key, ctx); */
void he_gemv(he_ct_t *ct_dest, const _Complex double *A, const he_ct_t *ct, const he_evk_t *rk)
{
  unsigned int slots = hectx.slots;
  /* plaintext A is of dim slots*slots.  */
  unsigned int n1 = sqrt(slots);
  if (slots!=n1*n1)
    n1 = sqrt(2*slots);       /* giant step */
  unsigned int n2 = slots/n1; /* baby  step */
  /* alloc */
  he_pt_t pt;
  he_alloc_pt(&pt);
  he_ct_t inner_sum, outer_sum, ct_rot;
  he_alloc_ct(&inner_sum);
  he_alloc_ct(&outer_sum);
  he_alloc_ct(&ct_rot);
  /* Compute sum */
  for (unsigned int i=0; i<n2; i++) {
    int shift = i*n1;
    for (unsigned int j=0; j<n1; j++) {
      /* rotate and encode ct */
      he_copy_ct(&ct_rot, ct);
      he_rot(&ct_rot, j, rk);
      /* rotate and encode A */
      _Complex double rotdiag[slots];
      zrotdiag(rotdiag, A, shift+j, -shift);
      he_ecd(&pt, rotdiag);
      /* multiply and accumulate */
      he_mulpt(&ct_rot, &ct_rot, &pt);
      if (!j)
        he_copy_ct(&inner_sum, &ct_rot);
      else
        he_add(&inner_sum, &inner_sum, &ct_rot);
    }
    he_rot(&inner_sum, shift, rk);
    if (!i)
      he_copy_ct(&outer_sum, &inner_sum);
    else
      he_add(&outer_sum, &outer_sum, &inner_sum);
  }
  he_copy_ct(ct_dest, &outer_sum);
  he_rs(ct_dest);
  /* release */
  he_free_pt(&pt);
  he_free_ct(&inner_sum);
  he_free_ct(&outer_sum);
  he_free_ct(&ct_rot);
}

void he_sum(he_ct_t *ct_sum, const he_ct_t *ct, const he_evk_t *rk)
{
  unsigned int slots = hectx.slots;
  _Complex double A[slots*slots];
  memset(A, 0, sizeof(A));
  for (unsigned int i=0; i<slots; i++)
    A[i]=1;
  he_gemv(ct_sum, A, ct, rk);
}

void he_idx(he_ct_t *ct_idx, const he_ct_t *ct, const unsigned int idx, const he_evk_t *rk)
{
  unsigned int slots = hectx.slots;
  _Complex double A[slots*slots];
  memset(A, 0, sizeof(A));
  A[idx*slots+idx]=1;
  he_gemv(ct_idx, A, ct, rk);
}

void he_nrm2(he_ct_t *ct_nrm2, const he_ct_t *ct, const he_evk_t *rlk, const he_evk_t *ck, const he_evk_t *rk)
{
  he_ct_t ct_conj;
  he_alloc_ct(&ct_conj);
  he_copy_ct(&ct_conj, ct);
  he_conj(&ct_conj, ck);
  he_mul(ct_nrm2, ct, &ct_conj, rlk);
  he_rs(ct_nrm2);
  he_sum(ct_nrm2, ct_nrm2, rk);
  he_free_ct(&ct_conj);
}

/**
 * he_inv - homomorphically evaluate inverse
 * 
 * Depth: iter+1
 */
void he_inv(he_ct_t *ct_inv, const he_ct_t *ct, const he_evk_t *rlk, const unsigned int iter)
{
  /* alloc */
  he_pt_t one, two;
  he_alloc_pt(&one);
  he_alloc_pt(&two);
  he_ct_t tmp, an, bn;
  he_alloc_ct(&tmp);
  he_alloc_ct(&an);
  he_alloc_ct(&bn);
  /* assign */
  he_const_pt(&one, 1);
  he_const_pt(&two, 2);
  /* main */
  he_copy_ct(&tmp, ct);
  he_neg(&tmp);              /* tmp = -ct */
  he_addpt(&an, &tmp, &two); /* an = 2-ct */
  he_moddown(&an);
  he_addpt(&bn, &tmp, &one); /* bn = 1-ct */
  for (unsigned int _=0; _<iter; _++) {
    he_mul(&bn, &bn, &bn, rlk);  /* bn = bn*bn */
    he_rs(&bn);
    he_addpt(&tmp, &bn, &one);   /* tmp = 1+bn */
    he_mul(&an, &an, &tmp, rlk);
    he_rs(&an);
  }
  he_copy_ct(ct_inv, &an);
  /* release */
  he_free_pt(&one);
  he_free_pt(&two);
  he_free_ct(&an);
  he_free_ct(&bn);
  he_free_ct(&tmp);
}

void he_sqrt(he_ct_t *ct_sqrt, const he_ct_t *ct, const he_evk_t *rlk, const unsigned int iter)
{
  /* alloc */
  he_pt_t one, three, half, quarter;
  he_alloc_pt(&one);
  he_alloc_pt(&three);
  he_alloc_pt(&half);
  he_alloc_pt(&quarter);
  he_ct_t tmp, an, bn;
  he_alloc_ct(&tmp);
  he_alloc_ct(&an);
  he_alloc_ct(&bn);
  /* assign */
  he_const_pt(&one, 1);
  he_const_pt(&three, 3);
  he_const_pt(&half, 0.5);
  he_const_pt(&quarter, 0.25);
  /* main */
  he_copy_ct(&an, ct);    /* an = ct */
  he_subpt(&bn, ct, &one); /* bn = ct-1 */
  for (unsigned int _=0; _<iter; _++) {
    /* an */
    he_mulpt(&tmp, &bn, &half);  /* bn/2 */
    he_rs(&tmp);
    he_subpt(&tmp, &tmp, &one);  /* bn/2-1 */
    he_neg(&tmp);                /* tmp = 1-bn/2 */
    he_moddown(&an);
    he_mul(&an, &an, &tmp, rlk); /* an = an*(1-bn/2) */
    he_rs(&an);
    /* bn */
    he_subpt(&tmp, &bn, &three);
    he_mulpt(&tmp, &tmp, &quarter); /* tmp = (bn-3)/4 */
    he_rs(&tmp);
    he_mul(&bn, &bn, &bn, rlk);     /* bn = bn*bn */
    he_rs(&bn);
    he_mul(&bn, &bn, &tmp, rlk);    /* bn = (bn*bn)*((bn-3)/4) */
    he_rs(&bn);
    he_show_ct_params(&an, "an[%u]", _);
  }
  he_copy_ct(ct_sqrt, &an);
}

void he_sigmoid(he_ct_t *ct_sigmoid, const he_ct_t *ct, const he_evk_t *rlk)
{
  /* alloc */
  he_ct_t ct2, ct4, ct8, ct3x, ct13, ct7x, ct57, ct9x;
  he_alloc_ct(&ct2);
  he_alloc_ct(&ct4);
  he_alloc_ct(&ct8);
  he_alloc_ct(&ct3x);
  he_alloc_ct(&ct13);
  he_alloc_ct(&ct7x);
  he_alloc_ct(&ct57);
  he_alloc_ct(&ct9x);
  he_pt_t pt;
  he_alloc_pt(&pt);
  /* ct2 = ct^2 */
  he_mul(&ct2, ct, ct, rlk);
  he_rs(&ct2); /* l-1 */
  /* ct4 = ct^4 */
  he_mul(&ct4, &ct2, &ct2, rlk);
  he_rs(&ct4); /* l-2 */
  /* ct8 = ct^8 */
  he_mul(&ct8, &ct4, &ct4, rlk);
  he_rs(&ct8); /* l-3 */
  /* ct3x */
  he_const_pt(&pt, -1./48);
  he_mulpt(&ct3x, ct, &pt);
  he_rs(&ct3x); /* l-1 */
  /* ct13 */
  he_const_pt(&pt, (1./4)/(-1./48));
  he_addpt(&ct13, &ct2, &pt);
  he_mul(&ct13, &ct3x, &ct13, rlk);
  he_rs(&ct13); /* l-2 */
  he_moddown(&ct13); /* l-3 */
  he_moddown(&ct13); /* l-4 */
  /* ct7x */
  he_const_pt(&pt, -17./80640);
  he_mulpt(&ct7x, ct, &pt);
  he_rs(&ct7x); /* l-1 */
  /* ct57 */
  he_const_pt(&pt, (1./480)/(-17./80640));
  he_addpt(&ct57, &ct2, &pt);
  he_mul(&ct57, &ct7x, &ct57, rlk);
  he_rs(&ct57); /* l-2 */
  he_mul(&ct57, &ct4, &ct57, rlk);
  he_rs(&ct57); /* l-3 */
  he_moddown(&ct57); /* l-4 */
  /* ct9x */
  he_const_pt(&pt, 31./1451520);
  he_mulpt(&ct9x, ct, &pt);
  he_rs(&ct9x); /* l-1 */
  he_moddown(&ct9x);
  he_moddown(&ct9x);
  he_mul(&ct9x, &ct9x, &ct8, rlk);
  he_rs(&ct9x);/* l-4 */
  /* sum all */
  he_add(ct_sigmoid, &ct13, &ct57);
  he_add(ct_sigmoid, ct_sigmoid, &ct9x);
  he_const_pt(&pt, 1./2);
  he_addpt(ct_sigmoid, ct_sigmoid, &pt);
  /* release */
  he_free_ct(&ct2);
  he_free_ct(&ct4);
  he_free_ct(&ct8);
  he_free_ct(&ct3x);
  he_free_ct(&ct13);
  he_free_ct(&ct7x);
  he_free_ct(&ct57);
  he_free_ct(&ct9x);
  he_free_pt(&pt);
}

void he_log(he_ct_t *ct_log, const he_ct_t *ct, const he_evk_t *rlk)
{
  /* alloc */
  he_ct_t ct2, ct4, ct8, ctodd, cteven, cttmp;
  he_alloc_ct(&ct2);
  he_alloc_ct(&ct4);
  he_alloc_ct(&ct8);
  he_alloc_ct(&ctodd);
  he_alloc_ct(&cteven);
  he_alloc_ct(&cttmp);
  he_pt_t pt;
  he_alloc_pt(&pt);
  /* ct2 = ct^2 */
  he_mul(&ct2, ct, ct, rlk);
  he_rs(&ct2); /* l-1 */
  /* ct4 = ct^4 */
  he_mul(&ct4, &ct2, &ct2, rlk);
  he_rs(&ct4); /* l-2 */
  /* ct8 = ct^8 */
  he_mul(&ct8, &ct4, &ct4, rlk);
  he_rs(&ct8);
  /* odd */
  he_copy_ct(&ctodd, &ct8);
  he_const_pt(&pt, 9./7.);
  he_mulpt(&cttmp, &ct2, &pt);
  he_rs(&cttmp); /* l-2 */
  he_mul(&cttmp, &cttmp, &ct4, rlk);
  he_rs(&cttmp); /* l-3 */
  he_add(&ctodd, &ctodd, &cttmp); /* (9/7)x^6+x^8 */
  he_const_pt(&pt, 9./5.);
  he_mulpt(&cttmp, &ct4, &pt);
  he_rs(&cttmp); /* l-3 */
  he_add(&ctodd, &ctodd, &cttmp); /* (9/5)x^4+(9/7)x^6+x^8 */
  he_const_pt(&pt, 9./3.);
  he_mulpt(&cttmp, &ct2, &pt);
  he_rs(&cttmp); /* l-2 */
  he_moddown(&cttmp); /* l-3 */
  he_add(&ctodd, &ctodd, &cttmp); /* (9/3)x^2+(9/5)x^4+(9/7)x^6+x^8 */
  he_const_pt(&pt, 9);
  he_addpt(&ctodd, &ctodd, &pt); /* 9+(9/3)x^2+(9/5)x^4+(9/7)x^6+x^8 */
  he_const_pt(&pt, 1./9.);
  he_mulpt(&cttmp, ct, &pt);
  he_rs(&cttmp); /* l-1 */
  he_moddown(&cttmp);
  he_moddown(&cttmp); /* l-3 */
  he_mul(&ctodd, &cttmp, &ctodd, rlk);
  he_rs(&ctodd); /* l-4 */
  /* even */
  he_copy_ct(&cteven, &ct8);
  he_const_pt(&pt, 10./8.);
  he_mulpt(&cttmp, &ct2, &pt);
  he_rs(&cttmp); /* l-2 */
  he_mul(&cttmp, &cttmp, &ct4, rlk);
  he_rs(&cttmp); /* l-3 */
  he_add(&cteven, &cteven, &cttmp); /* (10/8)x^6+x^8 */
  he_const_pt(&pt, 10./6.);
  he_mulpt(&cttmp, &ct4, &pt);
  he_rs(&cttmp); /* l-3 */
  he_add(&cteven, &cteven, &cttmp); /* (10/6)x^4+(10/8)x^6+x^8 */
  he_const_pt(&pt, 10./4.);
  he_mulpt(&cttmp, &ct2, &pt);
  he_rs(&cttmp); /* l-2 */
  he_moddown(&cttmp); /* l-3 */
  he_add(&cteven, &cteven, &cttmp); /* (10/4)x^2+(10/6)x^4+(10/8)x^6+x^8 */
  he_const_pt(&pt, 10./2);
  he_addpt(&cteven, &cteven, &pt); /* (10/2)+(10/4)x^2+(10/6)x^4+(10/8)x^6+x^8 */
  he_const_pt(&pt, -1./10.);
  he_mulpt(&cttmp, &ct2, &pt);
  he_rs(&cttmp); /* l-2 */
  he_moddown(&cttmp); /* l-3 */
  he_mul(&cteven, &cttmp, &cteven, rlk);
  he_rs(&cteven); /* l-4 */
  /* sum */
  he_add(ct_log, &ctodd, &cteven);
  /* release */
  he_free_ct(&ct2);
  he_free_ct(&ct4);
  he_free_ct(&ct8);
  he_free_ct(&ctodd);
  he_free_ct(&cteven);
  he_free_ct(&cttmp);
  he_free_pt(&pt);
}

/** exp(ct) */
static void he_exp_taylor(he_ct_t *ct_dest, const he_ct_t *ct, const he_evk_t *rlk)
{
  /* alloc */
  he_ct_t ct2, ct4, ct01, ct23, ct0123, ct45, ct67, ct4567;
  he_alloc_ct(&ct2);
  he_alloc_ct(&ct4);
  he_alloc_ct(&ct01);
  he_alloc_ct(&ct23);
  he_alloc_ct(&ct0123);
  he_alloc_ct(&ct45);
  he_alloc_ct(&ct67);
  he_alloc_ct(&ct4567);
  he_pt_t pt;
  he_alloc_pt(&pt);
  /* ct2 = ct^2 */
  he_mul(&ct2, ct, ct, rlk);
  he_rs(&ct2); /* l-1 */
  /* ct4 = ct^4 */
  he_mul(&ct4, &ct2, &ct2, rlk);
  he_rs(&ct4); /* l-2 */
  /* ct01 */
  he_const_pt(&pt, 1.0);
  he_addpt(&ct01, ct, &pt);
  he_mulpt(&ct01, &ct01, &pt);
  he_rs(&ct01); /* l-1 */
  he_moddown(&ct01); /* l-2 */
  /* ct23 */
  he_const_pt(&pt, 3.0);
  he_addpt(&ct23, ct, &pt);
  he_const_pt(&pt, 1.0/6); /* 1/3! */
  he_mulpt(&ct23, &ct23, &pt);
  he_rs(&ct23); /* l-1 */
  he_mul(&ct23, &ct2, &ct23, rlk);
  he_rs(&ct23); /* l-2 */
  /* ct0123 */
  he_add(&ct0123, &ct01, &ct23); /* l-2 */
  he_moddown(&ct0123); /* l-3 */
  /* ct45 */
  he_const_pt(&pt, 5.0);
  he_addpt(&ct45, ct, &pt);
  he_const_pt(&pt, 1.0/120); /* 1/5! */
  he_mulpt(&ct45, &ct45, &pt);
  he_rs(&ct45); /* l-1 */
  he_moddown(&ct45); /* l-2 */
  /* ct67 */
  he_const_pt(&pt, 7.0);
  he_addpt(&ct67, ct, &pt);
  he_const_pt(&pt, 1.0/5040); /* 1/7! */
  he_mulpt(&ct67, &ct67, &pt);
  he_rs(&ct67); /* l-1 */
  he_mul(&ct67, &ct2, &ct67, rlk);
  he_rs(&ct67); /* l-2 */
  /* ct4567 */
  he_add(&ct4567, &ct45, &ct67); /* l-2 */
  he_mul(&ct4567, &ct4, &ct4567, rlk);
  he_rs(&ct4567); /* l-3 */
  /* ct_dest */
  he_add(ct_dest, &ct0123, &ct4567); /* l-3 */
  /* release */
  he_free_ct(&ct2);
  he_free_ct(&ct4);
  he_free_ct(&ct01);
  he_free_ct(&ct23);
  he_free_ct(&ct0123);
  he_free_ct(&ct45);
  he_free_ct(&ct67);
  he_free_ct(&ct4567);
  he_free_pt(&pt);
}

/** exp(ct) */
void he_exp(he_ct_t *ct_exp,
            _Complex double a, const he_ct_t *ct,
            const he_evk_t *rlk, const unsigned int iter)
{
  unsigned int slots = hectx.slots;
  he_pt_t pt;
  he_alloc_pt(&pt);
  he_ct_t act;
  he_alloc_ct(&act);
  a /= (1<<iter);
  _Complex double coeff[slots];
  for (unsigned int i=0; i<slots; i++)
    coeff[i] = a;
  he_ecd(&pt, coeff);
  he_mulpt(&act, ct, &pt);
  he_rs(&act);
  he_exp_taylor(ct_exp, &act, rlk);
  for (unsigned int n=0; n<iter; n++) {
    he_mul(ct_exp, ct_exp, ct_exp, rlk);
    he_rs(ct_exp);
  }
  he_free_ct(&act);
  he_free_pt(&pt);
}

static void he_cmp_core(he_ct_t *an, const he_ct_t *ct, const he_evk_t *rlk,
  const unsigned int iter, const unsigned int t)
{
  he_ct_t bn, inv;
  he_alloc_ct(&bn);
  he_alloc_ct(&inv);
  he_copy_ct(&inv, an);
  he_pt_t one, half;
  he_alloc_pt(&one);
  he_alloc_pt(&half);
  he_const_pt(&one, 1);
  he_const_pt(&half, 0.5);
  /* inv((a+b)/2) */
  he_mulpt(&inv, &inv, &half);
  he_rs(&inv); /* l-1 */
  he_inv(&inv, &inv, rlk, iter); /* l-2-iter */
  /* a/2 */
  he_mulpt(an, ct, &half);
  he_rs(an); /* l-1 */
  for (unsigned int _=0; _<iter+1; _++)
    he_moddown(an);
  /* a0=(a/2)*inv((a+b)/2) */
  he_mul(an, an, &inv, rlk);
  he_rs(an); /* l-3-iter */
  /* b0=1-a0*/
  he_subpt(&bn, an, &one);
  he_neg(&bn);
  /* main */
  for (unsigned int n=0; n<t; n++) {
    he_mul(an, an, an, rlk);
    he_rs(an);
    he_mul(&bn, &bn, &bn, rlk);
    he_rs(&bn);
    he_add(&inv, an, &bn);
    he_inv(&inv, &inv, rlk, iter);
    for (unsigned int _=0; _<iter+1; _++)
      he_moddown(an);
    he_mul(an, an, &inv, rlk);
    he_rs(an);
    he_subpt(&bn, an, &one);
    he_neg(&bn);
  }
  /* release */
  he_free_pt(&one);
  he_free_pt(&half);
  he_free_ct(&bn);
  he_free_ct(&inv);
}

/**
 * he_cmp
 * 
 * Depth: (3+iter)*(1+t) = (3+iter)*(1+log2(alpha*2^alpha))
 */
void he_cmp(he_ct_t *ct_cmp, const he_ct_t *ct1, const he_ct_t *ct2, const he_evk_t *rlk,
            const unsigned int iter, const unsigned int alpha)
{
  /* iteration parameters */
  __attribute__((unused)) const unsigned int m = 2;
  const double c = 1+pow(2,-(int)alpha);
  const unsigned int t = log2(alpha/log2(c));
  /* alloc */
  he_ct_t an;
  he_alloc_ct(&an);
  /* ct1+ct2 */
  he_add(&an, ct1, ct2);
  he_cmp_core(&an, ct1, rlk, iter, t);
  he_copy_ct(ct_cmp, &an);
  /* release */
  he_free_ct(&an);
}

void he_cmppt(he_ct_t *ct_cmp, const he_ct_t *ct, const he_pt_t *pt, const he_evk_t *rlk,
  const unsigned int iter, const unsigned int alpha)
{
  /* iteration parameters */
  __attribute__((unused)) const unsigned int m = 2;
  const double c = 1+pow(2,-alpha);
  const unsigned int t = log2(alpha/log2(c));
  /* alloc */
  he_ct_t an;
  he_alloc_ct(&an);
  /* ct+pt */
  he_addpt(&an, ct, pt);
  he_cmp_core(&an, ct, rlk, iter, t);
  he_copy_ct(ct_cmp, &an);
  /* release */
  he_free_ct(&an);
}


END_DECLS
