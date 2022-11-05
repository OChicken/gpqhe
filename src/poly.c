/*
 * Polynomial arithmetic.
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
#include "kem.h"
#include "symmetric.h"

BEGIN_DECLS

extern MPI GPQHE_TWO;

/* poly.h */
extern struct poly_ctx polyctx;

/* kem.h */
extern struct kem_ctx kemctx;

/* reduce.c */
extern uint64_t barrett_reduce(u128 a, uint64_t q, uint64_t qinv);

/* ntt.c */
extern void    ntt(uint64_t a[], const struct rns_ctx *rns);
extern void invntt(uint64_t a[], const struct rns_ctx *rns);

/* rns.c */
extern void rns_decompose(uint64_t ahat[], const MPI a[], const struct rns_ctx *rns);
extern void rns_reconstruct(MPI a, const uint64_t ahat[], const unsigned int i, const struct rns_ctx *rns);

void poly_mpi_alloc(poly_mpi_t *a)
{
  a->coeffs = gcry_malloc(polyctx.n*sizeof(MPI));
  for (unsigned int i=0; i<polyctx.n; i++)
    a->coeffs[i] = gcry_mpi_new(0);
}

void poly_mpi_free(poly_mpi_t *a)
{
  for (unsigned int i=0; i<polyctx.n; i++)
    mpi_release(a->coeffs[i]);
  gcry_free(a->coeffs);
}

void poly_rns_alloc(poly_rns_t *a, const unsigned int dim)
{
  a->coeffs = malloc((dim*polyctx.n)*sizeof(uint64_t));
  memset(a->coeffs, 0, sizeof((dim*polyctx.n)*sizeof(uint64_t)));
}

void poly_rns_free(poly_rns_t *a)
{
  free(a->coeffs);
}

void poly_rns_add(uint64_t rhat[], const uint64_t ahat[], const uint64_t bhat[],
                  const struct rns_ctx *rns)
{
  for (unsigned int i=0; i<polyctx.n; i++)
    rhat[i] = barrett_reduce((u128)ahat[i]+(u128)bhat[i], rns->p, rns->pinv_barr);
}
void poly_rns_mul(uint64_t rhat[], const uint64_t ahat[], const uint64_t bhat[],
                  const struct rns_ctx *rns)
{
  for (unsigned int i=0; i<polyctx.n; i++)
    rhat[i] = barrett_reduce((u128)ahat[i]*(u128)bhat[i], rns->p, rns->pinv_barr);
}

void poly_mul(poly_mpi_t *r, const poly_mpi_t *a, const poly_mpi_t *b,
              const unsigned int dim, const MPI q)
{
  poly_rns_t rhat;
  poly_rns_alloc(&rhat, dim);
  poly_rns_t ahat;
  poly_rns_alloc(&ahat, 1);
  poly_rns_t bhat;
  poly_rns_alloc(&bhat, 1);
  struct rns_ctx *rns = polyctx.rns;
  for (unsigned int d=0; d<dim; d++) {
    rns_decompose(ahat.coeffs, a->coeffs, rns);
    rns_decompose(bhat.coeffs, b->coeffs, rns);
    ntt(ahat.coeffs, rns);
    ntt(bhat.coeffs, rns);
    poly_rns_mul(&rhat.coeffs[d*polyctx.n], ahat.coeffs, bhat.coeffs, rns);
    invntt(&rhat.coeffs[d*polyctx.n], rns);
    rns = (d<dim-1)? rns->next : rns;
  }
  poly_rns2mpi(r, &rhat, rns, q);
  poly_rns_free(&ahat);
  poly_rns_free(&bhat);
  poly_rns_free(&rhat);
}

void poly_rns2mpi(poly_mpi_t *r, const poly_rns_t *rhat,
                  const struct rns_ctx *rns, const MPI q)
{
  MPI qh = mpi_new(0);
  mpi_fdiv(qh, NULL, q, GPQHE_TWO);
  for (unsigned int i=0; i<polyctx.n; i++) {
    rns_reconstruct(r->coeffs[i], rhat->coeffs, i, rns);
    mpi_smod(r->coeffs[i], rns->P, rns->P_2);
    mpi_smod(r->coeffs[i], q, qh);
  }
  mpi_release(qh);
}

/**
 * poly_uniform - Sample a polynomial deterministically from a seed, with output
 * polynomial looking uniformly random.
 *
 * @poly_mpi_t *a: pointer to output polynomial
 * @seed[]:        pointer to input seed
 */
void poly_uniform(poly_rns_t *a, const unsigned char seed[], const uint64_t q)
{
  unsigned int ctr=0;
  uint16_t val;
  xof_state state;
  //uint64_t state[25];
  uint8_t buf[SHAKE128_RATE];
  uint8_t extseed[GPQHE_SYMBYTES+1];

  memcpy(extseed, seed, GPQHE_SYMBYTES);

  for(unsigned int i=0;i<polyctx.n/GPQHE_BLKSIZ;i++) { /* generate a in blocks of 64 coefficients */
    ctr = 0;
    extseed[GPQHE_SYMBYTES] = i; /* domain-separate the 16 independent calls */
    shake128_absorb(&state, extseed, GPQHE_SYMBYTES+1);
    while(ctr < 64) { /* Very unlikely to run more than once */
      shake128_squeezeblocks(buf,1,&state);
      for(unsigned int j=0;j<SHAKE128_RATE && ctr < 64;j+=2) {
        val = (buf[j] | ((uint16_t) buf[j+1] << 8));
        if(val < 5*q) {
          a->coeffs[i*64+ctr] = val;
          ctr++;
        }
      }
    }
  }
}

/**
 * hw - Compute the Hamming weight of an unsigned char use shift.
 * 
 * @a: The input char.
 * Return: The Hamming weight of char a.
 */
static inline uint8_t hw(uint8_t a)
{
  a -= (a >> 1) & 0x55;
  a = (a & 0x33) + ((a >> 2) & 0x33);
  return (a + (a >> 4)) & 0x0F;
}

/**
 * load32_littleendian - load 4 bytes into a 32-bit integer in little-endian order.
 *
 * @x: pointer to input byte array
 * @return: 32-bit unsigned integer loaded from x
 */
static inline uint32_t load32_littleendian(const uint8_t x[4])
{
  uint32_t r;
  r  = (uint32_t)x[0];
  r |= (uint32_t)x[1] << 8;
  r |= (uint32_t)x[2] << 16;
  r |= (uint32_t)x[3] << 24;
  return r;
}

/**
 * cbd - centered binomial distribution
 *
 * @x[4]: pointer to input byte array
 * @return: the cbd corresponding number
 */
static inline uint32_t cbd(const uint8_t x[4])
{
  uint32_t t = load32_littleendian(x);
  uint32_t d = 0;
  for (unsigned int k=0; k<8; k++)
    d += (t>>k) & 0x01010101;
  return d;
}

void poly_sample(poly_mpi_t *r, const unsigned char *seed, unsigned char nonce)
{
  unsigned char buf[GPQHE_BLKSIZ*2];
//  uint32_t t, d, a, b, c;
  MPI a = mpi_new(0);
  MPI b = mpi_new(0);
  MPI c = mpi_new(0);
  MPI d = mpi_new(0);

  uint8_t extkey[GPQHE_SYMBYTES+2];
  memcpy(extkey, seed, GPQHE_SYMBYTES);
  extkey[GPQHE_SYMBYTES] = nonce;

  for(unsigned int i=0; i<polyctx.n/GPQHE_BLKSIZ; i++) { /* Generate noise in blocks of 64 coefficients */
    extkey[GPQHE_SYMBYTES+1] = i;
    shake256(buf, sizeof(buf), extkey, GPQHE_SYMBYTES+2);
    //prf(buf, sizeof(buf), extkey, GPQHE_SYMBYTES+1);
    for(unsigned int j=0; j<GPQHE_BLKSIZ; j++) {
      //a = buf[2*j];
      //b = buf[2*j+1];
      //r->coeffs[64*i+j] = hw(a) + NEWHOPE_Q - hw(b);
      mpi_set_ui(a, hw(buf[2*j  ]));
      mpi_set_ui(b, hw(buf[2*j+1]));
      mpi_set(r->coeffs[GPQHE_BLKSIZ*i+j], polyctx.q);
      //mpi_set_ui(r->coeffs[GPQHE_BLKSIZ*i+j], 0);
      mpi_add(r->coeffs[GPQHE_BLKSIZ*i+j], r->coeffs[GPQHE_BLKSIZ*i+j], a);
      mpi_sub(r->coeffs[GPQHE_BLKSIZ*i+j], r->coeffs[GPQHE_BLKSIZ*i+j], b);
      uint32_t s = cbd(buf+j);
      mpi_set_ui(a,  s     &0xff);
      mpi_set_ui(b, (s>> 8)&0xff);
      mpi_set_ui(c, (s>>16)&0xff);
      mpi_set_ui(d, (s>>24)&0xff);
      /* r->coeffs[64*i+j/2  ] = a+q-b */
      mpi_set(r->coeffs[GPQHE_BLKSIZ*i+j/2  ], polyctx.q);
      //mpi_set_ui(r->coeffs[GPQHE_BLKSIZ*i+j/2  ], 0);
      mpi_add(r->coeffs[GPQHE_BLKSIZ*i+j/2  ], r->coeffs[GPQHE_BLKSIZ*i+j/2  ], a);
      mpi_sub(r->coeffs[GPQHE_BLKSIZ*i+j/2  ], r->coeffs[GPQHE_BLKSIZ*i+j/2  ], b);
      /* r->coeffs[64*i+j/2+1] = c+q-d */
      mpi_set(r->coeffs[GPQHE_BLKSIZ*i+j/2+1], polyctx.q);
      //mpi_set_ui(r->coeffs[GPQHE_BLKSIZ*i+j/2+1], 0);
      mpi_add(r->coeffs[GPQHE_BLKSIZ*i+j/2+1], r->coeffs[GPQHE_BLKSIZ*i+j/2+1], c);
      mpi_sub(r->coeffs[GPQHE_BLKSIZ*i+j/2+1], r->coeffs[GPQHE_BLKSIZ*i+j/2+1], d);
      /*
      t = buf[j] | ((uint32_t)buf[j+1] << 8) | ((uint32_t)buf[j+2] << 16) | ((uint32_t)buf[j+3] << 24);
      d = 0;
      for(k=0;k<8;k++)
        d += (t >> k) & 0x01010101;
      a = d & 0xff;
      b = ((d >>  8) & 0xff);
      c = ((d >> 16) & 0xff);
      d >>= 24;
      r->coeffs[64*i+j/2]   = a + NEWHOPE_Q - b;
      r->coeffs[64*i+j/2+1] = c + NEWHOPE_Q - d;
      */
    }
  }
  mpi_release(a);
  mpi_release(b);
  mpi_release(c);
  mpi_release(d);
}

void poly_rot(poly_mpi_t *r, const poly_mpi_t *a, size_t rot)
{
  size_t power = 1;
  size_t m = 2*polyctx.n;
  for (unsigned int j=0; j<rot; j++)
    power *= GPQHE_ROT; /* power = pow(5,rot) */
  for (unsigned int i=0; i<polyctx.n; i++) {
    unsigned int k = (i*power)%m;
    if (k<polyctx.n)
      mpi_set(r->coeffs[k], a->coeffs[i]); /* r[k] = x[i] */
    else
      mpi_neg(r->coeffs[k-polyctx.n], a->coeffs[i]); /* r[k-n] = -x[i] */
  }
}

void poly_conj(poly_mpi_t *r, const poly_mpi_t *a)
{
  mpi_set(r->coeffs[0], a->coeffs[0]);
  for (unsigned int i=1; i<polyctx.n; i++)
    mpi_neg(r->coeffs[i], a->coeffs[polyctx.n-i]);
}

#if 0
struct mult_arg{
  uint64_t *rhat;
  uint64_t *ahat;
  uint64_t *bhat;
  MPI *r;
  MPI *a;
  MPI *b;
  unsigned int k;
};

static void *_mult(void *arg)
{
  struct mult_arg *dat = arg;
  unsigned int k = dat->k;
  MPI a = mpi_new(0);
  MPI b = mpi_new(0);
  MPI q = mpi_new(0);
  mpi_set_ui(q, ctx.p[k]);
  for (unsigned int i=0; i<ctx.n; i++) {
    mpi_mod(a, dat->a[i], q);
    dat->ahat[i] = mpi_to_u64(a);
    mpi_mod(b, dat->b[i], q);
    dat->bhat[i] = mpi_to_u64(b);
  }
  mpi_release(a);
  mpi_release(b);
  mpi_release(q);
  ntt(dat->ahat, k);
  ntt(dat->bhat, k);
  for (unsigned int i=0; i<ctx.n; i++) {
    dat->rhat[i] = barrett_reduce((u128)(dat->ahat[i])*(u128)(dat->bhat[i]),
      ctx.p[k], ctx.pinv_barr[k]);
  }
  invntt(dat->rhat, k);
#ifdef USE_THREAD
  pthread_exit(NULL);
#else
  return NULL;
#endif
}

void poly_mulm(MPI r[], MPI a[], MPI b[],
  const MPI q, const unsigned int np)
{
  uint64_t rhat[np*ctx.n];
  uint64_t ahat[np*ctx.n];
  uint64_t bhat[np*ctx.n];
  struct mult_arg arg;
  for (unsigned int k=0; k<np; k++) {
    arg.k = k;
    arg.rhat = &rhat[k*ctx.n];
    arg.ahat = &ahat[k*ctx.n];
    arg.bhat = &bhat[k*ctx.n];
    arg.a = a;
    arg.b = b;
    _mult(&arg);
    memcpy(&rhat[k*ctx.n], arg.rhat, ctx.n*sizeof(uint64_t));
  }
  invcrt(r, rhat, np);
  for (unsigned int i=0; i<ctx.n; i++)
    mpi_mod(r[i], r[i], q);
}

#ifndef USE_THREAD
void poly_ntt_to_mpi(poly_mpi_t *r, poly_ntt_t *rhat, const unsigned int np)
{
  MPI a = mpi_new(0);
  MPI c = mpi_new(0);
  for (unsigned int i=0; i<ctx.n; i++) {
    mpi_set_ui(r->coeffs[i], 0);
    for (unsigned int k=0; k<np; k++) {
      mpi_set_ui(a, rhat->coeffs[i][k]);
      mpi_set_ui(c, ctx.phat_invmp[np-1][k]);             /* c = phat^{-1}mod pi */
      mpi_mulm(c, ctx.phat[np-1][k], c, ctx.P[np-1]); /* c = (phat*c) mod P  */
      mpi_mulm(a, a, c, ctx.P[np-1]);                 /* a[k] = a[k]*c mod P */
      mpi_add(r->coeffs[i], r->coeffs[i], a);
      mpi_mod(r->coeffs[i], r->coeffs[i], ctx.P[np-1]);
    }
  }
  mpi_release(a);
  mpi_release(c);
}
#else
#endif
#endif
END_DECLS
