/*
 * Residual number system.
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

BEGIN_DECLS

/* poly.h */
extern struct poly_ctx polyctx;

/**
 * rns_decompose - decompose the polynomial in mpi domain to its RNS domain,
 * whose axis info is stored in `rns`.
 * 
 * @ahat: polynomial in the d-th axis of RNS domain (d=0,1,...,dim-1)
 * @a: polynomial in mpi domain
 * @rns: the RNS parameters of the d-th axis of RNS domain, whose size (P) is
 * sufficient large to accommodate the polynomial modulus in mpi domain.
 */
void rns_decompose(uint64_t ahat[], const MPI a[], const struct rns_ctx *rns)
{
  MPI b = mpi_new(0);
  MPI q = mpi_new(0);
  mpi_set_ui(q, rns->p);
  for (unsigned int i=0; i<polyctx.n; i++) {
    mpi_mod(b, a[i], q);
    ahat[i] = mpi_to_u64(b);
  }
  mpi_release(b);
  mpi_release(q);
}

/**
 * rns_reconstruct - reconstruct the i-th coefficient of the polynomial in mpi
 * domain from its RNS domain.
 * 
 * @a: the i-th coefficient of the polynomial in mpi domain
 * @ahat: polynomial in RNS domain
 * @i: polynomial coefficient index
 * @rns: the RNS parameters of the polynomial modulus, whose size (P) is 
 * sufficient large to accommodate the polynomial modulus in mpi domain.
 */
void rns_reconstruct(MPI a, const uint64_t ahat[],
                     const unsigned int i, const struct rns_ctx *rns)
{
  mpi_set_ui(a, 0);
  MPI b = mpi_new(0);
  MPI c = mpi_new(0);
  for (unsigned int d=0; d<rns->dim; d++) {
    mpi_set_ui(b, ahat[d*polyctx.n+i]);
    mpi_set_ui(c, rns->phat_invmp[d]);    /* c = phat^{-1}mod pj */
    mpi_mulm(c, rns->phat[d], c, rns->P); /* c = (phat*c) mod P  */
    mpi_mulm(b, b, c, rns->P);            /* a[i] = a[i]*c mod P */
    mpi_addm(a, a, b, rns->P);
  }
  mpi_release(c);
  mpi_release(b);
}

END_DECLS

#if 0
struct crt_arg{
  uint64_t *xhat;
  MPI *x;
  unsigned int k;
};

static void *_crt(void *arg)
{
  struct crt_arg *dat = arg;
  unsigned int k = dat->k;
  MPI a = mpi_new(0);
  MPI q = mpi_new(0);
  mpi_set_ui(q, ctx.p[k]);
  for (unsigned int i=0; i<GPQHE_N; i++) {
    mpi_mod(a, dat->x[i], q);
    dat->xhat[i] = mpi_to_u64(a);
  }
  mpi_release(a);
  mpi_release(q);
#ifdef USE_THREAD
  pthread_exit(NULL);
#else
  return NULL;
#endif
}

void crt(uint64_t xhat[], MPI x[GPQHE_N], const unsigned int np)
{
#ifdef USE_THREAD
  unsigned int end = np;
  for (unsigned int start=0; start<end; start+=GPQHE_NUM_THREAD) {
    unsigned int num_thread = end-start;
    num_thread = (num_thread>GPQHE_NUM_THREAD)? GPQHE_NUM_THREAD : num_thread;
    struct crt_arg arg[num_thread];
    pthread_t tids[num_thread];
    int rc = 0;
    void *status;
    for (unsigned int k=0; k<num_thread; k++) {
      arg[k].k = start+k;
      arg[k].xhat = &xhat[arg[k].k*GPQHE_N];
      arg[k].x = x;
      rc |= pthread_create(&tids[k], NULL, _crt, (void*)&arg[k]);
    }
    for (unsigned int k=0; k<num_thread; k++) {
      rc |= pthread_join(tids[k], &status);
      memcpy(&xhat[arg[k].k*GPQHE_N], arg[k].xhat, GPQHE_N*sizeof(uint64_t));
    }
    assert(!rc);
    assert(status==NULL);
  }
#else
  struct crt_arg arg;
  for (unsigned int k=0; k<np; k++) {
    arg.k = k;
    arg.xhat = &xhat[k*GPQHE_N];
    arg.x = x;
    _crt(&arg);
    memcpy(&xhat[k*GPQHE_N], arg.xhat, GPQHE_N*sizeof(uint64_t));
  }
#endif
}

struct invcrt_arg{
  MPI x;
  uint64_t *xhat;
  unsigned int i;
  unsigned int np;
};

static void *_invcrt(void *arg)
{
  struct invcrt_arg *dat = arg;
  unsigned int i = dat->i;
  unsigned int np = dat->np;
  MPI a = mpi_new(0);
  MPI c = mpi_new(0);
  for (unsigned int k=0; k<np; k++) {
    mpi_set_ui(a, dat->xhat[k*GPQHE_N+i]);
    mpi_set_ui(c, ctx.phat_invmp[np-1][k]);             /* c = phat^{-1}mod pi */
    mpi_mulm(c, ctx.phat[np-1][k], c, ctx.P[np-1]); /* c = (phat*c) mod P  */
    mpi_mulm(a, a, c, ctx.P[np-1]);                 /* a[k] = a[k]*c mod P */
    mpi_addm(dat->x, dat->x,a, ctx.P[np-1]);
  }
  mpi_release(c);
  mpi_release(a);
  if (mpi_cmp(dat->x, ctx.P_2[np-1])>0)
    mpi_sub(dat->x, dat->x, ctx.P[np-1]);
#ifdef USE_THREAD
  pthread_exit(NULL);
#else
  return NULL;
#endif
}

void invcrt(MPI x[GPQHE_N], uint64_t xhat[], const unsigned int np)
{
#ifdef USE_THREAD
  unsigned int end = GPQHE_N;
  for (unsigned int start=0; start<end; start+=GPQHE_NUM_THREAD) {
    unsigned int num_thread = end-start;
    num_thread = (num_thread>GPQHE_NUM_THREAD)? GPQHE_NUM_THREAD : num_thread;
    struct invcrt_arg arg[num_thread];
    pthread_t tids[num_thread];
    int rc = 0;
    void *status;
    for (unsigned int i=0; i<num_thread; i++) {
      arg[i].i = start+i;
      arg[i].np = np;
      arg[i].xhat = xhat;
      arg[i].x = mpi_new(0);
      mpi_set_ui(arg[i].x, 0);
      rc |= pthread_create(&tids[i], NULL, _invcrt, (void*)&arg[i]);
    }
    for (unsigned int i=0; i<num_thread; i++) {
      rc |= pthread_join(tids[i], &status);
      mpi_set(x[arg[i].i], arg[i].x);
      mpi_release(arg[i].x);
    }
    assert(!rc);
    assert(status==NULL);
  }
#else
  struct invcrt_arg arg;
  arg.xhat  = xhat;
  arg.np    = np;
  arg.x  = mpi_new(0);
  for (unsigned int i=0; i<GPQHE_N; i++) {
    arg.i = i;
    mpi_set_ui(arg.x, 0);
    _invcrt(&arg);
    mpi_set(x[i], arg.x);
  }
  mpi_release(arg.x);
#endif
}

#endif
