/*
 * Precompute constants.
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
#include "gpqhe.h"
#include <stdbool.h>
#include <stdio.h>        /* fprintf */
#include <stdlib.h>       /* malloc, abort, rand */
#include <string.h>       /* memset, strcpy, strerror */
#include <math.h>
#include <complex.h>

/* error */
#include <assert.h>
#include <errno.h>

BEGIN_DECLS

/* types.c */
MPI GPQHE_TWO;

/* polyctx.h */
struct poly_ctx polyctx;

/* kem.h */
struct kem_ctx kemctx;

/* fhe.h */
struct he_ctx hectx;

/* reduce.c */
extern uint64_t montgomery_inv(uint64_t q);
extern uint64_t barrett_inv(uint64_t q);

static unsigned int he_std_params(unsigned int logn)
{
#if (GPQHE_CQ == 'C')
#if (GPQHE_SEC_LEVEL == 128)   /* 128 bits classical security */
  switch (logn) {
    case 10: return  27;
    case 11: return  54;
    case 12: return 109;
    case 13: return 218;
    case 14: return 438;
    case 15: return 881;
  }
#elif (GPQHE_SEC_LEVEL == 192) /* 192 bits classical security */
  switch (logn) {
    case 10: return  19;
    case 11: return  37;
    case 12: return  75;
    case 13: return 152;
    case 14: return 305;
    case 15: return 611;
  }
#elif (GPQHE_SEC_LEVEL == 256) /* 256 bits classical security */
  switch (logn) {
    case 10: return  14;
    case 11: return  29;
    case 12: return  58;
    case 13: return 118;
    case 14: return 237;
    case 15: return 476;
  }
#endif
/* end of `if (GPQHE_CQ == 'C')` */
#elif (GPQHE_CQ == 'Q')
#if (GPQHE_SEC_LEVEL == 128)   /* 128 bits quantum security */
  switch (logn) {
    case 10: return  25;
    case 11: return  51;
    case 12: return 101;
    case 13: return 202;
    case 14: return 411;
    case 15: return 827;
  }
#elif (GPQHE_SEC_LEVEL == 192) /* 192 bits quantum security */
  switch (logn) {
    case 10: return  17;
    case 11: return  35;
    case 12: return  70;
    case 13: return 141;
    case 14: return 284;
    case 15: return 571;
  }
#elif (GPQHE_SEC_LEVEL == 256) /* 256 bits quantum security */
  switch (logn) {
    case 10: return  13;
    case 11: return  27;
    case 12: return  54;
    case 13: return 109;
    case 14: return 220;
    case 15: return 443;
  }
#endif
/* end of `if (GPQHE_CQ == 'Q')` */
#endif
  return 0;
}

/** powm - power mod: a^b mod m */
static uint64_t powm(uint64_t a, uint64_t b, uint64_t m)
{
  uint64_t r=1;
  while(b>0) {
    if (b&1)
      r = (uint64_t)( ((u128)r*(u128)a) % (u128)m );
    b>>=1;
    a =   (uint64_t)( ((u128)a*(u128)a) % (u128)m );
  }
  return r;
}

/** bit-reversal operation to a 32-bit integer */
static uint32_t bitrev_u32(uint32_t a)
{
  a = ((a & 0xaaaaaaaa)>> 1) | ((a & 0x55555555)<< 1);
  a = ((a & 0xcccccccc)>> 2) | ((a & 0x33333333)<< 2);
  a = ((a & 0xf0f0f0f0)>> 4) | ((a & 0x0f0f0f0f)<< 4);
  a = ((a & 0xff00ff00)>> 8) | ((a & 0x00ff00ff)<< 8);
  return (a>>16)|(a<<16);
}

/**
 * isprime - Miller-Rabin prime test with 50 witness attempts.
 * 
 * @p: The uint64 input number.
 * 
 * The function returns true if p is prime, and false if p is composite.
 * The function aborts and set errno=EINVAL if p=0 or p=1.
 * 
 * "50 witness attempts" is suggested in <Introduction to Algorithms> by Thomas
   H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein.
 */
static bool isprime(uint64_t p)
{
  const size_t num_of_prime_test = 50;
  bool ret = false;
  if (p<2) {
    errno = EINVAL;
    fprintf(stderr, "\033[1m\033[31merror:\033[0m \033[1m%s\033[0m. "
      "The input %lu is neither prime nor composite.\n", strerror(errno), p);
    abort();
  }
  else if (p==2)  /* p>=2 */
    return true;
  else if (!(p&1)) /* p>2, p&1==0 */
    return false;
  else /* we are certain that p is odd, so p-1 is even --- contains factor 2.*/
  {
    uint64_t u=p-1;
    uint64_t t=0;
    while (!(u&1))
      { u/=2; t++; }
    for (size_t k=0; k<num_of_prime_test; k++) {
      uint64_t a = rand();
      a = (a<<32)|rand();
      a = a%(p-1)+1;
      uint64_t x=powm(a,u,p);
      for (uint64_t i=1; i<=t; i++) {
        uint64_t y = (uint64_t)( ((u128)x*(u128)x) % (u128)p );
        if ( (y==1) && (x!=1) && (x!=p-1))
          return false;
        x = y;
      }
      if (x!=1)
        return false;
      else
        ret = true;
    }
  }
  return ret;
}

/** factorize `N` and store the result in `factors`. */
static void factorize(uint64_t *factors, uint64_t N)
{
  size_t i=0;
  for (uint64_t n=2; n<=__builtin_sqrt(N); n++){
    while (N%n==0)
      { factors[i++]=n; N/=n; }
  }
  if (N>2)
    factors[i++]=N;
}

/** get generator of PRIME p */
static uint64_t generator(uint64_t p)
{
  uint64_t phi = p-1;
  size_t pbits = 64UL-__builtin_clzll(p);
  /* factorize */
  uint64_t faphi[pbits];
  memset(faphi, 0, sizeof(faphi));
  factorize(faphi, phi);
  /* generator */
  uint64_t g;
  for (g=2; g<=phi; g++) {
    bool flag = 0;
    for (size_t j=0; faphi[j]; j++) {
      if (powm(g, phi/faphi[j], p)==1)
        { flag = 1; break; }
    }
    if (flag==0)
      break;
  }
  return g;
}

/**
 * m-th root of unity
 *
 * @m: the order
 * @p: prime
 * return a which satisfies a^m == 1 mod p
 */
static uint64_t mth_root_of_unity(uint64_t m, uint64_t p)
{
  uint64_t phi = p-1;
  uint64_t g = generator(p);
  /* m-th root of unity */
  assert(phi%m==0);
  return powm(g, phi/m, p);
}

static void ntt_init(uint64_t p)
{
  polyctx.rns->pinv_mont = montgomery_inv(p);
  polyctx.rns->pinv_barr = barrett_inv(p);
  polyctx.rns->ninv      = ( ( (u128)(powm(polyctx.n, p-2, p)) * polyctx.R ) % (u128)(p) );
  polyctx.rns->zetas     = malloc(polyctx.n*sizeof(uint64_t));
  polyctx.rns->zetas_inv = malloc(polyctx.n*sizeof(uint64_t));
  uint64_t root = mth_root_of_unity(2*polyctx.n, p);
  uint64_t rootinv = powm(root, p-2, p);
  uint64_t power = 1;
  uint64_t powerInv = 1;
  for (unsigned int i=0; i<polyctx.n; i++) {
    uint32_t j = bitrev_u32(i) >> (32-polyctx.logn);
    uint64_t rootpow = power;
    uint64_t rootpowInv = powerInv;
    polyctx.rns->zetas[j]     = ( (u128)rootpow    * polyctx.R ) % p;
    polyctx.rns->zetas_inv[j] = ( (u128)rootpowInv * polyctx.R ) % p;
    power    = (u128)power * (u128)root % (u128)p;
    powerInv = (u128)powerInv * (u128)rootinv % (u128)p;
  }
}

static void rns_init(struct rns_ctx **origin, const MPI Pprev)
{
  MPI pd                  = mpi_new(0);
  MPI phat_d              = mpi_new(0);
  polyctx.rns->P          = mpi_new(0);
  polyctx.rns->P_2        = mpi_new(0);
  polyctx.rns->phat       = gcry_malloc(polyctx.rns->dim*sizeof(MPI));
  polyctx.rns->phat_invmp = malloc(polyctx.rns->dim*sizeof(MPI));
  if (polyctx.rns->dim==1)
    mpi_set_ui(polyctx.rns->P, polyctx.rns->p);
  else
    mpi_mul_ui(polyctx.rns->P, Pprev, polyctx.rns->p);
  mpi_fdiv(polyctx.rns->P_2, NULL, polyctx.rns->P, GPQHE_TWO);
  struct rns_ctx *rns;
  for (unsigned int d=0; d<polyctx.rns->dim; d++) {
    if (d==0)
      rns = *origin;
    else
      rns = rns->next;
    polyctx.rns->phat[d] = mpi_new(0);
    mpi_set_ui(pd, rns->p);
    mpi_fdiv(polyctx.rns->phat[d], NULL, polyctx.rns->P, pd); /* P/pd */
    mpi_mod(phat_d, polyctx.rns->phat[d], pd); /* phat_d = (P/pd) mod pd */
    polyctx.rns->phat_invmp[d] = powm(mpi_to_u64(phat_d), rns->p-2, rns->p); /* invm(phat%p, p) = phat^{-1} mod pd */
  }
  mpi_release(pd);
  mpi_release(phat_d);
}

static void ring_init(const unsigned int n, const unsigned int m)
{
  /* `nh` means n half */
  unsigned int nh = n/2;
  /* cyclic group */
  polyctx.ring.cyc_group = malloc(nh*sizeof(unsigned int));
  polyctx.ring.cyc_group[0] = 1;
  for (unsigned int i=1; i<nh; i++)
    polyctx.ring.cyc_group[i] = (GPQHE_ROT*polyctx.ring.cyc_group[i-1]) % m;
  /* powers of each primitive roots until the number of slots. */
  polyctx.ring.zetas = malloc((m+1)*sizeof(_Complex double));
  for (size_t i=0; i<m; i++) {
    double theta = 2*GPQHE_PI*i / m;
    polyctx.ring.zetas[i] = cos(theta) + I*sin(theta);
  }
  polyctx.ring.zetas[m] = polyctx.ring.zetas[0];
#if 0
  /* ring constang cM */
  polyctx.ring.cM = 0;
  for (unsigned int i=0; i<n; i++) {
    double sum_row = 0;
    for (unsigned int j=0; j<nh; j++) {
      double theta = 2*GPQHE_PI*polyctx.ring.cyc_group[j]/m;
      _Complex double zeta = cos(theta) + I*sin(theta);
      sum_row += creal(pow(zeta,i));
    }
    sum_row = 2*sum_row/n;
    if (polyctx.ring.cM<sum_row)
      polyctx.ring.cM = sum_row;
  }
#endif
}

void polyctx_init(unsigned int logn, MPI q)
{
  GPQHE_TWO = mpi_new(0);
  mpi_set_ui(GPQHE_TWO, 2);
  /* polyctxnomial degree */
  polyctx.logn = logn;
  polyctx.n    = (1u << polyctx.logn);
  polyctx.m    = 2*polyctx.n;
  /* modulus q */
  polyctx.logq = mpi_get_nbits(q)-1;
  polyctx.logqub = he_std_params(polyctx.logn);
  if ((logn<10)|| (15<logn))
    polyctx.logqub = polyctx.logq; /* dedicated for KAT of HEAAN and/or personal designed parameters */
  /* Comment the following for internal test. Bear in mind that there is a constraint 
    upon degree n and modulus q, q should not arbitrary large to guarantee security. */
#if 1
  if (polyctx.logq > polyctx.logqub) {
    errno = EINVAL;
    fprintf(stderr, "\033[1m\033[31merror:\033[0m \033[1m%s\033[0m. "
      "The input modulus q is too large. "
      "Must guarantee log(q)<=log(qub).\n", strerror(errno));
    abort();
  }
#endif
  polyctx.q = mpi_new(0);
  mpi_set(polyctx.q, q);
  polyctx.logR  = BITS_PER_LONG;
  polyctx.R     = (u128)1 << BITS_PER_LONG;
  polyctx.Rsub1 = polyctx.R-1;
  polyctx.dimub = (1 + polyctx.logn + 4*polyctx.logqub) / GPQHE_LOGP + 1;
  uint64_t p = (1ULL<<GPQHE_LOGP) + 1;
  struct rns_ctx *origin = NULL;
  MPI Pprev = mpi_new(0);
  for (unsigned int dim=1; dim<=polyctx.dimub; dim++) {
    if (dim==1) {
      polyctx.rns = malloc(sizeof(struct rns_ctx));
      origin = polyctx.rns;
    }
    else {
      polyctx.rns->next = malloc(sizeof(struct rns_ctx));
      mpi_set(Pprev, polyctx.rns->P);
      polyctx.rns = polyctx.rns->next;
    }
    polyctx.rns->dim = dim;
    while(true) {
      p += 2*polyctx.n;
      if(isprime(p))
        { polyctx.rns->p = p; break; }
    }
    ntt_init(p);
    rns_init(&origin, Pprev);
    polyctx.rns->next = NULL;
  }
  ring_init(polyctx.n, polyctx.m);
  polyctx.rns = origin;
  mpi_release(Pprev);
}

static void qtable_init(const MPI qL)
{
  MPI q = mpi_copy(qL);
  unsigned int logq = mpi_get_nbits(q)-1;
  unsigned int logDelta = mpi_get_nbits(hectx.p)-1;
  hectx.L  = ceil(logq/logDelta);
  hectx.q  = gcry_malloc((hectx.L+1)*sizeof(MPI));
  hectx.qh = gcry_malloc((hectx.L+1)*sizeof(MPI));
  for (int l=hectx.L; l>=0; l--) {
    hectx.q[l]  = mpi_new(0);
    hectx.qh[l] = mpi_new(0);
    mpi_set(hectx.q[l], q);
    mpi_fdiv(hectx.qh[l], NULL, q, GPQHE_TWO);
    mpi_fdiv(q, NULL, q, hectx.p);
  }
  hectx.dim = (mpi_get_nbits(hectx.q[hectx.L])+polyctx.logn)/GPQHE_LOGP+1;
  struct rns_ctx *rns=polyctx.rns;
  for (unsigned int d=0; d<hectx.dim-1; d++, rns=rns->next);
  hectx.P   = mpi_copy(rns->P);
  hectx.PqL = mpi_new(0);
  mpi_mul(hectx.PqL, hectx.P, qL);
  hectx.dimevk = (mpi_get_nbits(hectx.q[hectx.L])+mpi_get_nbits(hectx.PqL)+polyctx.logn)/GPQHE_LOGP+1;
  mpi_release(q);
}

static void bounds_init(const unsigned int n)
{
  unsigned int h = GPQHE_BLKSIZ; /* hamming weight */
  double sigma = GPQHE_SIGMA;
  hectx.bnd.Bclean = 8*sqrt(2)*sigma*n + 6*sigma*sqrt(n) + 16*sigma*sqrt(h*n);
  hectx.bnd.Brs = sqrt(n/3.) * (3+8*sqrt(h));
  hectx.bnd.Bks = 8*sigma*n/sqrt(3);
  hectx.bnd.Bmult = malloc((hectx.L+1)*sizeof(double));
  long double Pinvql[hectx.L+1];
  long double Pinv = 1;
  for (struct rns_ctx *rns=polyctx.rns; rns; rns=rns->next)
    Pinv *= 1./rns->p;
  Pinvql[0] = Pinv * mpi_to_u64(hectx.q[0]);
  hectx.bnd.Bmult[0] = Pinvql[0]*hectx.bnd.Bks + hectx.bnd.Brs;
  for (unsigned int l=1; l<=hectx.L; l++) {
    Pinvql[l] = Pinvql[l-1]*hectx.Delta;
    hectx.bnd.Bmult[l] = Pinvql[l]*hectx.bnd.Bks + hectx.bnd.Brs;
  }
}

void hectx_init(unsigned int logn, MPI q, unsigned int slots, uint64_t Delta)
{
  polyctx_init(logn, q);
  if ((slots&(slots-1))!=0) {
    errno = EINVAL;
    fprintf(stderr, "\033[1m\033[31merror:\033[0m \033[1m%s\033[0m. "
      "The slots must be the power of 2.\n", strerror(errno));
    abort();
  }
  if (slots>(polyctx.n/2)) {
    errno = EINVAL;
    fprintf(stderr, "\033[1m\033[31merror:\033[0m \033[1m%s\033[0m. "
      "Must guarantee slots<=(n/2).\n", strerror(errno));
    abort();
  }
  hectx.slots = slots;
  hectx.Delta = (double)Delta;
  hectx.p = mpi_set_ui(NULL, Delta);
  qtable_init(q);
  bounds_init(polyctx.n);
  /* bounds */
  assert(Delta > (polyctx.n + 2*hectx.bnd.Bclean));
}

void kemctx_init(unsigned int ssbytes){
  kemctx.polybytes = (polyctx.logq+1)*polyctx.n/8;
  kemctx.pkbytes   = kemctx.polybytes + GPQHE_SYMBYTES;
  kemctx.skbytes   = kemctx.polybytes;
  if (hectx.slots)
    kemctx.ssbytes = hectx.slots*(polyctx.n/2)*sizeof(_Complex double);
  else
    kemctx.ssbytes  = ssbytes;
}

void polyctx_exit()
{
  mpi_release(GPQHE_TWO);
  mpi_release(polyctx.q);
  for (unsigned int dim=1; dim<=polyctx.dimub; dim++) {
    struct rns_ctx *p=polyctx.rns->next;
    free(polyctx.rns->zetas);
    polyctx.rns->zetas=NULL;
    free(polyctx.rns->zetas_inv);
    polyctx.rns->zetas_inv=NULL;
    mpi_release(polyctx.rns->P);
    mpi_release(polyctx.rns->P_2);
    for (unsigned int d=0; d<dim; d++)
      mpi_release(polyctx.rns->phat[d]);
    gcry_free(polyctx.rns->phat);
    free(polyctx.rns->phat_invmp);
    free(polyctx.rns);
    polyctx.rns=NULL;
    polyctx.rns=p;
  }
  free(polyctx.ring.cyc_group);
  free(polyctx.ring.zetas);
}

void hectx_exit()
{
  polyctx_exit();
  for (unsigned int l=0; l<=hectx.L; l++) {
    mpi_release(hectx.q[l]);
    mpi_release(hectx.qh[l]);
  }
  gcry_free(hectx.q);
  gcry_free(hectx.qh);
  mpi_release(hectx.p);
  mpi_release(hectx.P);
  mpi_release(hectx.PqL);
  free(hectx.bnd.Bmult);
}

END_DECLS
