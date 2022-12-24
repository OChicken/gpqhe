/*
 * Test file: homomorphic encryption, decryption, addition and multiplication.
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

#include "../src/gpqhe.h"
#include "../pmu/pmu.h"
#include <complex.h>
#include <math.h>

/* fhe.h */
extern struct he_ctx hectx;

/* sample.c */
extern void sample_z01vec(_Complex double vec[], const unsigned int m);

/* canemb.c */
extern void canemb(_Complex double a[], const unsigned int slots);
extern void invcanemb(_Complex double a[], const unsigned int slots);

/* encode.c */
extern double canemb_norm(const _Complex double m[], const double Delta);
extern double canemb_norm_(const poly_mpi_t *a);

static inline double
blas_dznrmmax(const _Complex double z[], const unsigned int len)
{
  double norm = 0;
  for (unsigned int i=0; i<len; i++) {
    double tmp = cabs(z[i]);
    if (norm<tmp)
      norm = tmp;
  }
  return norm;
}

static inline double
blas_dnrmmax_dist(const double y[], const double x[], const unsigned int len)
{
  double dist = 0;
  for (unsigned int i=0; i<len; i++) {
    double tmp = fabs(y[i] - x[i]);
    if (dist<tmp)
      dist = tmp;
  }
  return dist;
}

static inline double
blas_dznrmmax_dist(const _Complex double y[], const _Complex double x[], const unsigned int len)
{
  double dist = 0;
  for (unsigned int i=0; i<len; i++) {
    double tmp = cabs(y[i] - x[i]);
    if (dist<tmp)
      dist = tmp;
  }
  return dist;
}

static inline void
blas_dzrot(_Complex double y[], const _Complex double x[], const unsigned int len, const int rot)
{
  assert(y!=x);
  for (unsigned int i=0; i<len; i++) {
    int idx = ((int)i+rot)%len;
    if (idx<0)
      idx += len;
    assert(idx>=0);
    y[i] = x[idx];
  }
}

static inline void
blas_dzinv(_Complex double r[], const _Complex double x[], const unsigned int len, const unsigned int iter)
{
  for (unsigned int i=0; i<len; i++) {
    _Complex double an, bn;
    an = 2-x[i];
    bn = 1-x[i];
    for (unsigned int _=0; _<iter; _++) {
      bn = bn*bn;
      an = an*(bn+1);
    }
    r[i] = an;
  }
}

static inline void
blas_dinv(double r[], const double x[], const unsigned int len, const unsigned int iter)
{
  for (unsigned int i=0; i<len; i++) {
    double an, bn;
    an = 2-x[i];
    bn = 1-x[i];
    for (unsigned int _=0; _<iter; _++) {
      bn = bn*bn;
      an = an*(bn+1);
    }
    r[i] = an;
  }
}

static inline void
comp(_Bool cmp[], const _Complex double a0[], const _Complex double b0[], const unsigned int len,
  const unsigned int d1, const unsigned int d2, const int alpha)
{
  /* iteration parameters */
  const unsigned int m = 2;
  const double c = 1+pow(2,-alpha);
  const unsigned int t = log2(alpha/log2(c));
  double a[len], b[len], inv[len];
  for (unsigned int i=0; i<len; i++)
    a[i] = (a0[i]+b0[i])/2;
  blas_dinv(a, a, len, d2);
  for (unsigned int i=0; i<len; i++) {
    a[i] *= (a0[i])/2;
    b[i] = 1-a[i];
  }
  for (unsigned int _=0; _<t; _++) {
    for (unsigned int i=0; i<len; i++)
      inv[i] = pow(a[i], m)+pow(b[i], m);
    blas_dinv(inv, inv, len, d1);
    for (unsigned int i=0; i<len; i++) {
      a[i] = pow(a[i], m)*inv[i];
      b[i] = 1-a[i];
    }
  }
  for (unsigned int i=0; i<len; i++)
    cmp[i] = (unsigned int)round(creal(a[i]));
}

static unsigned int logn;
static unsigned int logq;
static unsigned int logDelta;
static char key[2];
static unsigned int slots;
static uint64_t Delta;
static poly_mpi_t sk;
static he_pk_t pk;
static he_evk_t rlk;
static he_evk_t ck;
static he_pt_t pt;
static he_ct_t ct;
static unsigned int idx;
static unsigned int start;
static unsigned int end;
static unsigned int alpha;
static double diff;
static unsigned int iter;

#define CHECK_DIFF(m,m0)                       \
  TEST_PLACEHOLDER();                          \
  diff = blas_dznrmmax_dist((m), (m0), slots); \
  printf("diff = %g\n", diff);                 \
  ASSERT(diff<1e-5);

static void test_ecd()
{
TEST_BEGIN();
  he_pt_t pt1, pt2;
  he_alloc_pt(&pt);
  he_alloc_pt(&pt1);
  he_alloc_pt(&pt2);
  _Complex double m[slots], m0[slots], m1[slots], m2[slots];

TEST_DO("canemb correctness");
  sample_z01vec(m0, slots);
  memcpy(m, m0, sizeof(m0));
  invcanemb(m,slots);
  canemb(m,slots);
  CHECK_DIFF(m, m0);
TEST_DONE();

TEST_DO("encode/decode");
  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  he_dcd(m, &pt);
  CHECK_DIFF(m, m0);
TEST_DONE();

TEST_DO("canemb_norm");
  sample_z01vec(m1, slots);
  sample_z01vec(m2, slots);
  for (unsigned int i=0; i<slots; i++) {
    m1[i] = (1-(double)i/slots)+(1-(double)i/slots)*I;
    m2[i] = (0.99-(double)i/slots)+(0.99-(double)i/slots)*I;
  }
  for (unsigned int i=0; i<slots; i++)
    m0[i] = m1[i]*m2[i];
  he_ecd(&pt1, m1);
  he_ecd(&pt2, m2);
  if (hectx.Delta>canemb_norm(m1, hectx.Delta))
    printf("Delta > canemb_norm_(m1, Delta)\n");
  else
    printf("Delta < canemb_norm_(m1, Delta)\n");
  if (hectx.Delta>canemb_norm_(&pt1.m))
    printf("Delta > canemb_norm(pt1)\n");
  else
    printf("Delta < canemb_norm(pt1)\n");
  if (hectx.Delta>canemb_norm(m2, hectx.Delta))
    printf("Delta > canemb_norm_(m2, Delta)\n");
  else
    printf("Delta < canemb_norm_(m2, Delta)\n");
  if (hectx.Delta>canemb_norm_(&pt2.m))
    printf("Delta > canemb_norm(pt2)\n");
  else
    printf("Delta < canemb_norm(pt2)\n");
  printf("pt1.nu                    = %f\n", pt1.nu);
  printf("canemb_norm(m1, Delta)    = %f\n", canemb_norm(m1, hectx.Delta));
  printf("canemb_norm(pt1)          = %f\n", canemb_norm_(&pt1.m));
  printf("pt2.nu                    = %f\n", pt2.nu);
  printf("canemb_norm(m2, Delta)    = %f\n", canemb_norm(m2, hectx.Delta));
  printf("canemb_norm(pt2)          = %f\n", canemb_norm_(&pt2.m));
  unsigned int dim = (mpi_get_nbits(hectx.q[hectx.L])*2)/GPQHE_LOGP+1;
  poly_mul(&pt.m, &pt1.m, &pt2.m, dim, hectx.q[hectx.L]);
  pt.nu = pt1.nu*pt2.nu;
  printf("pt.nu                     = %f\n", pt.nu);
  printf("canemb_norm_(m0, Delta^2) = %f\n", canemb_norm(m0, (double)hectx.Delta*hectx.Delta));
  printf("canemb_norm(pt)           = %f\n", canemb_norm_(&pt.m));
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

  he_free_pt(&pt);
  he_free_pt(&pt1);
  he_free_pt(&pt2);
TEST_END();
}

static void test_enc()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_alloc_pk(&pk);
  he_alloc_ct(&ct);
  he_alloc_sk(&sk);
  _Complex double m[slots], m0[slots];
  sample_z01vec(m, slots);
  memcpy(m0, m, sizeof(m));

TEST_DO("encode");
  he_ecd(&pt, m);
TEST_DONE();
  he_show_pt_params(&pt, "pt before encrypt");

TEST_DO("encrypt");
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);
TEST_DONE();
  he_show_ct_params(&ct, "ct encrypted by %s", key);

TEST_DO("decrypt");
  he_dec(&pt, &ct, &sk);
TEST_DONE();
  he_show_pt_params(&pt, "pt after decrypt");

TEST_DO("decode");
  he_dcd(m, &pt);
TEST_DONE();
  CHECK_DIFF(m,m0);

TEST_DO("moddown");
  he_moddown(&ct);
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("exit");
  he_free_pt(&pt);
  he_free_ct(&ct);
TEST_DONE();

TEST_END();
}

static void test_add()
{
TEST_BEGIN();
  he_pt_t pt1, pt2;
  he_alloc_pt(&pt);
  he_alloc_pt(&pt1);
  he_alloc_pt(&pt2);
  he_ct_t ct1, ct2;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct1);
  he_alloc_ct(&ct2);

  _Complex double m[slots], m0[slots], m1[slots], m2[slots];
  sample_z01vec(m1, slots);
  sample_z01vec(m2, slots);

TEST_DO("enc(m1)+enc(m2)");
  for (unsigned int i=0; i<slots; i++)
    m0[i] = m1[i]+m2[i];
  he_ecd(&pt1, m1);
  he_ecd(&pt2, m2);
  if (!strcmp(key, "sk")) {
    he_enc_sk(&ct1, &pt1, &sk);
    he_enc_sk(&ct2, &pt2, &sk);
  }
  if (!strcmp(key, "pk")) {
    he_enc_pk(&ct1, &pt1, &pk);
    he_enc_pk(&ct2, &pt2, &pk);
  }
  he_add(&ct, &ct1, &ct2);
  he_show_ct_params(&ct1, "ct1 (%s)", key);
  he_show_ct_params(&ct2, "ct2 (%s)", key);
  he_show_ct_params(&ct, "ct=ct1+ct2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("enc(m1)+m2");
  for (unsigned int i=0; i<slots; i++)
    m0[i] = m1[i]+m2[i];
  he_ecd(&pt1, m1);
  he_ecd(&pt2, m2);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct1, &pt1, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct1, &pt1, &pk);
  he_addpt(&ct, &ct1, &pt2);
  he_show_ct_params(&ct1, "ct1 (%s)", key);
  he_show_pt_params(&pt2, "pt2");
  he_show_ct_params(&ct, "ct=ct1+pt2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("enc(m1)-enc(m2)");
  for (unsigned int i=0; i<slots; i++)
    m0[i] = m1[i]-m2[i];
  he_ecd(&pt1, m1);
  he_ecd(&pt2, m2);
  if (!strcmp(key, "sk")) {
    he_enc_sk(&ct1, &pt1, &sk);
    he_enc_sk(&ct2, &pt2, &sk);
  }
  if (!strcmp(key, "pk")) {
    he_enc_pk(&ct1, &pt1, &pk);
    he_enc_pk(&ct2, &pt2, &pk);
  }
  he_sub(&ct, &ct1, &ct2);
  he_show_ct_params(&ct1, "ct1 (%s)", key);
  he_show_ct_params(&ct2, "ct2 (%s)", key);
  he_show_ct_params(&ct, "ct=ct1-ct2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("enc(m1)-m2");
  for (unsigned int i=0; i<slots; i++)
    m0[i] = m1[i]-m2[i];
  he_ecd(&pt1, m1);
  he_ecd(&pt2, m2);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct1, &pt1, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct1, &pt1, &pk);
  he_subpt(&ct, &ct1, &pt2);
  he_show_ct_params(&ct1, "ct1 (%s)", key);
  he_show_pt_params(&pt2, "pt2");
  he_show_ct_params(&ct, "ct=ct1-pt2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("m1-enc(m2)");
  for (unsigned int i=0; i<slots; i++)
    m0[i] = m1[i]-m2[i];
  he_ecd(&pt1, m1);
  he_ecd(&pt2, m2);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct2, &pt2, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct2, &pt2, &pk);
  he_subpt(&ct, &ct2, &pt1);
  he_neg(&ct);
  he_show_pt_params(&pt1, "pt1");
  he_show_ct_params(&ct2, "ct2 (%s)", key);
  he_show_ct_params(&ct, "ct=pt1-ct2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("exit");
  he_free_pt(&pt);
  he_free_pt(&pt1);
  he_free_pt(&pt2);
  he_free_ct(&ct);
  he_free_ct(&ct1);
  he_free_ct(&ct2);
TEST_DONE();

TEST_END();
}

static void test_mul()
{
TEST_BEGIN();
  he_pt_t pt1, pt2;
  he_alloc_pt(&pt);
  he_alloc_pt(&pt1);
  he_alloc_pt(&pt2);
  he_ct_t ct1, ct2;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct1);
  he_alloc_ct(&ct2);
  he_alloc_evk(&rlk);
  _Complex double m[slots], m0[slots], m1[slots], m2[slots];
  sample_z01vec(m1, slots);
  sample_z01vec(m2, slots);

  for (unsigned int i=0; i<slots; i++)
    m0[i] = m1[i]*m2[i];
  he_ecd(&pt1, m1);
  he_ecd(&pt2, m2);
  if (!strcmp(key, "sk")) {
    he_enc_sk(&ct1, &pt1, &sk);
    he_enc_sk(&ct2, &pt2, &sk);
  }
  if (!strcmp(key, "pk")) {
    he_enc_pk(&ct1, &pt1, &pk);
    he_enc_pk(&ct2, &pt2, &pk);
  }
  he_show_ct_params(&ct1, "ct1 (%s)", key);
  he_show_ct_params(&ct2, "ct2 (%s)", key);

TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();

TEST_DO("enc(m1)*enc(m2) without rescale");
  he_mul(&ct, &ct1, &ct2, &rlk);
  he_show_ct_params(&ct, "ct=ct1*ct2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("enc(m1)*enc(m2)");
  he_mul(&ct, &ct1, &ct2, &rlk);
  he_show_ct_params(&ct, "ct=ct1*ct2");
  he_rs(&ct);
  he_show_ct_params(&ct, "ct=ct1*ct2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("enc(m1)*enc(m2) moddown correctness");
  he_mul(&ct, &ct1, &ct2, &rlk);
  he_rs(&ct);
  he_moddown(&ct);
  he_show_ct_params(&ct, "ct=ct1*ct2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("enc(m1)*enc(m2) moddown correctness with ct1 ct2 are moddown");
  he_moddown(&ct1);
  he_moddown(&ct2);
  he_show_ct_params(&ct1, "ct1 (%s)", key);
  he_show_ct_params(&ct2, "ct2 (%s)", key);
  he_mul(&ct, &ct1, &ct2, &rlk);
  he_rs(&ct);
  he_show_ct_params(&ct, "ct=ct1*ct2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

  if (!strcmp(key, "sk"))
    he_enc_sk(&ct1, &pt1, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct1, &pt1, &pk);

TEST_DO("enc(m1)*m2");
  he_mulpt(&ct, &ct1, &pt2);
  he_rs(&ct);
  he_show_ct_params(&ct1, "ct1 (%s)", key);
  he_show_pt_params(&pt2, "pt2");
  he_show_ct_params(&ct, "ct=ct1*pt2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("enc(m1)*m2 with ct1 is moddown");
  he_moddown(&ct1);
  he_mulpt(&ct, &ct1, &pt2);
  he_rs(&ct);
  he_show_ct_params(&ct1, "ct1 (%s)", key);
  he_show_pt_params(&pt2, "pt2");
  he_show_ct_params(&ct, "ct=ct1*ct2");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("exit");
  he_free_evk(&rlk);
  he_free_pt(&pt);
  he_free_pt(&pt1);
  he_free_pt(&pt2);
  he_free_ct(&ct);
  he_free_ct(&ct1);
  he_free_ct(&ct2);
TEST_DONE();

TEST_END();
}

static void test_conj()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct0;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct0);
  he_alloc_evk(&ck);
  _Complex double m[slots], m0[slots];
  sample_z01vec(m0, slots);
  he_ecd(&pt, m0);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct0, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct0, &pt, &pk);
  he_show_ct_params(&ct0, "ct0 (%s)", key);
  memcpy(m, m0, sizeof(m0));
  for (unsigned int i=0; i<slots; i++)
    m0[i] = conj(m0[i]);

TEST_DO("gen ck");
  he_genck(&ck, &sk);
TEST_DONE();

  he_copy_ct(&ct, &ct0);
TEST_DO("conj(m)");
  he_conj(&ct, &ck);
  he_show_ct_params(&ct, "conj(ct)");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

  he_copy_ct(&ct, &ct0);
  he_moddown(&ct);
  he_moddown(&ct);
TEST_DO("conj(m) with ckhat with ct has been moddown");
  he_conj(&ct, &ck);
  he_show_ct_params(&ct, "conj(ct)");
  he_dec(&pt, &ct, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m,m0);
TEST_DONE();

TEST_DO("exit");
  he_free_evk(&ck);
  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct0);
TEST_DONE();

TEST_END();
}

static void test_rot()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct0;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct0);
  he_evk_t rk[hectx.slots];
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_alloc_evk(&rk[rot]);
  _Complex double m[slots], m0[slots], mr[slots];
  sample_z01vec(m0, slots);
  he_ecd(&pt, m0);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct0, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct0, &pt, &pk);
  he_show_ct_params(&ct0, "ct0 (%s)", key);

TEST_DO("gen rk");
  he_genrk(rk, &sk);
TEST_DONE();

TEST_DO("rot(m)");
  for (unsigned int rot=0; rot<hectx.slots; rot++) {
    blas_dzrot(mr, m0, slots, rot);
    he_copy_ct(&ct, &ct0);
    he_rot(&ct, rot, rk);
    he_dec(&pt, &ct, &sk);
    he_dcd(m, &pt);
    TEST_PLACEHOLDER();
    diff = blas_dznrmmax_dist(m, mr, slots);
    printf("diff[%d] = %g\n", rot, diff);
    ASSERT(diff<1e-5);
  }
TEST_DONE();

TEST_DO("exit");
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_free_evk(&rk[rot]);
  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct0);
TEST_DONE();

TEST_END();
}

static void test_gemv()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_Av, ct_v;
  he_alloc_ct(&ct_Av);
  he_alloc_ct(&ct_v);
  he_evk_t rk[hectx.slots];
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_alloc_evk(&rk[rot]);
  _Complex double m[slots], Av[slots], A[slots*slots], v[slots];
  sample_z01vec(v, slots);
  sample_z01vec(A, slots*slots);
  memset(Av, 0, sizeof(Av));
  for (unsigned int i=0; i<slots; i++)
    for (unsigned int j=0; j<slots; j++)
      Av[i] += A[i*slots+j]*v[j];
  he_ecd(&pt, v);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct_v, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct_v, &pt, &pk);

TEST_DO("gen rk");
  he_genrk(rk, &sk);
TEST_DONE();

TEST_DO("gemv");
  he_gemv(&ct_Av, A, &ct_v, rk);
  he_dec(&pt, &ct_Av, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m, Av);
TEST_DONE();

TEST_DO("exit");
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_free_evk(&rk[rot]);
  he_free_pt(&pt);
  he_free_ct(&ct_v);
  he_free_ct(&ct_Av);
TEST_DONE();

TEST_END();
}

static void test_sum()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_sum;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_sum);
  he_evk_t rk[hectx.slots];
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_alloc_evk(&rk[rot]);
  _Complex double m[slots], m0[slots], sum=0, sum0=0;
  sample_z01vec(m0, slots);
  for (unsigned int i=0; i<slots; i++)
    sum0 += m0[i];

TEST_DO("gen rk");
  he_genrk(rk, &sk);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

TEST_DO("sum");
  he_sum(&ct_sum, &ct, rk);
  he_dec(&pt, &ct_sum, &sk);
  he_dcd(m, &pt);
  sum = m[0];
  diff = cabs(sum-sum0);
  TEST_PLACEHOLDER();
  printf("diff = %g\n", diff);
  ASSERT(diff<1e-5);
TEST_DONE();

TEST_DO("exit");
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_free_evk(&rk[rot]);
  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct_sum);
TEST_DONE();

TEST_END();
}

static void test_idx()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_sum;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_sum);
  he_evk_t rk[hectx.slots];
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_alloc_evk(&rk[rot]);
  _Complex double m[slots], m0[slots];
  sample_z01vec(m0, slots);

TEST_DO("gen rk");
  he_genrk(rk, &sk);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

TEST_DO("idx=1");
  he_idx(&ct_sum, &ct, idx, rk);
  he_dec(&pt, &ct_sum, &sk);
  he_dcd(m, &pt);
  diff = cabs(m[idx]-m0[idx]);
  TEST_PLACEHOLDER();
  printf("diff = %g\n", diff);
  ASSERT(diff<1e-5);
TEST_DONE();

TEST_DO("exit");
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_free_evk(&rk[rot]);
  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct_sum);
TEST_DONE();

TEST_END();
}

static void test_nrm2()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_nrm2;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_nrm2);
  he_alloc_evk(&rlk);
  he_alloc_evk(&ck);
  he_evk_t rk[hectx.slots];
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_alloc_evk(&rk[rot]);
  _Complex double m[slots], m0[slots], sum=0, sum0=0;
  sample_z01vec(m0, slots);
  for (unsigned int i=0; i<slots; i++)
    sum0 += m0[i]*conj(m0[i]);

TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();
TEST_DO("gen ck");
  he_genck(&ck, &sk);
TEST_DONE();
TEST_DO("gen rk");
  he_genrk(rk, &sk);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

TEST_DO("nrm2");
  he_nrm2(&ct_nrm2, &ct, &rlk, &ck, rk);
  he_show_ct_params(&ct_nrm2, "ct_nrm2");
  he_dec(&pt, &ct_nrm2, &sk);
  he_dcd(m, &pt);
  sum = m[0];
  diff = cabs(sum-sum0);
  TEST_PLACEHOLDER();
  printf("diff = %g\n", diff);
  ASSERT(diff<1e-5);
  TEST_PLACEHOLDER();
  printf("norm2=(%f,%f)\n", creal(sum), cimag(sum));
TEST_DONE();

TEST_DO("exit");
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_free_evk(&rk[rot]);
  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_evk(&rlk);
  he_free_evk(&ck);
  he_free_ct(&ct_nrm2);
TEST_DONE();

TEST_END();
}

static void test_inv()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_inv;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_inv);
  he_alloc_evk(&rlk);
  _Complex double m[slots], m0[slots], minv[slots];
  sample_z01vec(m0, slots);
  for (unsigned int i=0; i<slots; i++) {
    m0[i] = creal(m0[i])+0.5;
    minv[i] = 1/m0[i];
  }

TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
TEST_DO("raw");
  blas_dzinv(m, m0, slots, iter);
  CHECK_DIFF(m, minv);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

TEST_DO("inv");
  he_inv(&ct_inv, &ct, &rlk, iter);
  he_show_ct_params(&ct_inv, "ct_inv");
  he_dec(&pt, &ct_inv, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m, minv);
TEST_DONE();

  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct_inv);
  he_free_evk(&rlk);

TEST_END();
}

static void test_exp()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_exp;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_exp);
  he_alloc_evk(&rlk);
  _Complex double m[slots], m0[slots], mexp[slots];
  _Complex double a = 2*GPQHE_PI*I/Delta;
  sample_z01vec(m0, slots);
  for (unsigned int i=0; i<slots; i++) {
    m0[i] *= a;
    mexp[i] = cexp(m0[i]);
  }

TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

TEST_DO("exp(2*pi*m/Delta)");
  he_exp(&ct_exp, a, &ct, &rlk, iter);
  he_show_ct_params(&ct_exp, "ct_exp");
  he_dec(&pt, &ct_exp, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m, mexp);
TEST_DONE();

  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct_exp);
  he_free_evk(&rlk);

TEST_END();
}

static void test_sigmoid()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_sigmoid;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_sigmoid);
  he_alloc_evk(&rlk);
  _Complex double m[slots], m0[slots], msigmoid[slots];
  sample_z01vec(m0, slots);
TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();

TEST_DO("1/(1+e^(-x)) (complex)");
  for (unsigned int i=0; i<slots; i++) {
    m0[i] /=10;
    msigmoid[i] = 1/(1+cexp(-m0[i]));
  }

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

  he_sigmoid(&ct_sigmoid, &ct, &rlk);
  he_show_ct_params(&ct_sigmoid, "ct_sigmoid");
  he_dec(&pt, &ct_sigmoid, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m, msigmoid);
TEST_DONE();

  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct_sigmoid);
  he_free_evk(&rlk);

TEST_END();
}

static void test_log()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_log;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_log);
  he_alloc_evk(&rlk);
  _Complex double m[slots], m0[slots], mraw[slots], mlog[slots];
  sample_z01vec(m0, slots);
TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();

TEST_DO("log(x), x near 1");
  for (unsigned int i=0; i<slots; i++) {
    m0[i] = creal(m0[i])/100000;
    double x = creal(m0[i]);
    mlog[i] = log(1+creal(m0[i]));
    mraw[i] = (x/9)*(9+(9./3)*pow(x,2)+(9./5)*pow(x,4)+(9./7)*pow(x,6)+pow(x,8))
  +(-x*x/10)*((10./2)+(10./4)*pow(x,2)+(10./6)*pow(x,4)+(10./8)*pow(x,6)+pow(x,8));
    printf("%.10e %.10e \n", creal(m0[i]), creal(mlog[i]));
  }
  CHECK_DIFF(mraw, mlog);

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

  he_log(&ct_log, &ct, &rlk);
  he_show_ct_params(&ct_log, "ct_log");
  he_dec(&pt, &ct_log, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m, mlog);
  CHECK_DIFF(m, mraw);
TEST_DONE();

  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct_log);
  he_free_evk(&rlk);

TEST_END();
}

static void test_cmp()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct1, ct2, ct_cmp;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct1);
  he_alloc_ct(&ct2);
  he_alloc_ct(&ct_cmp);
  he_alloc_evk(&rlk);
#if 1
TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();
#endif

  _Complex double m[slots], m0[slots], ma[slots], mb[slots];
  sample_z01vec(m0, slots);
  sample_z01vec(m0, slots);
  for (unsigned int i=0; i<slots; i++) {
    ma[i] = creal(m0[i])+0.5;
    mb[i] = cimag(m0[i])+0.5;
  }
  printf("a = ");
  for (unsigned int i=0; i<slots; i++)
    printf("%f ", creal(ma[i]));
  printf("\n");
  printf("b = ");
  for (unsigned int i=0; i<slots; i++)
    printf("%f ", creal(mb[i]));
  printf("\n");

  memcpy(m, m0, sizeof(m0));
TEST_DO("raw");
  _Bool cmp[slots];
  comp(cmp, ma, mb, slots, iter, iter, alpha);
  for (unsigned int i=0; i<slots; i++)
    ASSERT(cmp[i]==((creal(ma[i])>creal(mb[i]))? 1:0));
TEST_DONE();

TEST_DO("cmp");
  he_ecd(&pt, ma);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct1, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct1, &pt, &pk);
  he_ecd(&pt, mb);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct2, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct2, &pt, &pk);
  he_cmp(&ct_cmp, &ct1, &ct2, &rlk, iter, alpha);
  he_show_ct_params(&ct_cmp, "ct_cmp");
  he_dec(&pt, &ct_cmp, &sk);
  he_dcd(m, &pt);
  printf("cmp(a,b) = ");
  for (unsigned int i=0; i<slots; i++) {
    printf("%f ", creal(m[i]));
    cmp[i] = round(creal(m[i]));
  }
  printf("\n");
  for (unsigned int i=0; i<slots; i++)
    ASSERT(cmp[i]==((creal(ma[i])>creal(mb[i]))? 1:0));
TEST_DONE();

  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct1);
  he_free_ct(&ct2);
  he_free_ct(&ct_cmp);
  he_free_evk(&rlk);

TEST_END();
}

static void test_coeff2slot()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct0, ct1;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct0);
  he_alloc_ct(&ct1);
  he_alloc_evk(&ck);
  he_evk_t rk[hectx.slots];
  for (unsigned int rot=0; rot<hectx.slots; rot++)
    he_alloc_evk(&rk[rot]);
  _Complex double m[slots], m0[slots], mr[slots], mi[slots], mr0[slots], mi0[slots];
  sample_z01vec(m0, slots);
  for (unsigned int i=0; i<slots; i++)
    m0[i] /= Delta;
  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  /* also encode, but not send to mpi */
  invcanemb(m, hectx.slots);
  for (unsigned int i=0; i<slots; i++) {
    mr0[i] = creal(m[i])*Delta;
    mi0[i] = cimag(m[i])*Delta;
  }
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

  he_bootstrapctx_init();

TEST_DO("gen ck");
  he_genck(&ck, &sk);
TEST_DONE();

TEST_DO("gen rk");
  he_genrk(rk, &sk);
TEST_DONE();

#if 1
TEST_DO("coefficient to slots");
  he_coeff2slot(&ct0, &ct1, &ct, &ck, rk);
  he_dec(&pt, &ct0, &sk);
  he_dcd(mr, &pt);
  he_dec(&pt, &ct1, &sk);
  he_dcd(mi, &pt);
  CHECK_DIFF(mr, mr0);
  CHECK_DIFF(mi, mi0);
TEST_DONE();
#endif

  he_bootstrapctx_exit();
  printf("%s %u\n", __func__, __LINE__);
  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct0);
  he_free_ct(&ct1);
TEST_END();
}

static void test_rlsin()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_new;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_new);
  he_alloc_evk(&rlk);
  he_alloc_evk(&ck);
  _Complex double m[slots], m0[slots];
  _Complex double a = 2*GPQHE_PI*I/Delta;
  sample_z01vec(m0, slots);
  for (unsigned int i=0; i<slots; i++)
    m0[i] /= Delta;

TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();

TEST_DO("gen ck");
  he_genck(&ck, &sk);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

TEST_DO("(1/(2*pi))*sin(2*pi*m/Delta)");
  he_rlsin(&ct_new, a, &ct, &rlk, &ck, iter);
  he_show_ct_params(&ct_new, "ct_new");
  he_dec(&pt, &ct_new, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m, m0);
TEST_DONE();

  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct_new);
  he_free_evk(&rlk);
  he_free_evk(&ck);
TEST_END();
}

static void test_sqrt()
{
TEST_BEGIN();
  he_alloc_pt(&pt);
  he_ct_t ct_sqrt;
  he_alloc_ct(&ct);
  he_alloc_ct(&ct_sqrt);
  he_alloc_evk(&rlk);
  _Complex double m[slots], m0[slots], msqrt[slots];
  sample_z01vec(m0, slots);
  for (unsigned int i=0; i<slots; i++) {
    m0[i] = creal(m0[i]);
    msqrt[i] = sqrt(creal(m0[i]));
  }

TEST_DO("gen rlk");
  he_genrlk(&rlk, &sk);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
TEST_DO("raw");
  for (unsigned int i=0; i<slots; i++){
    _Complex double an, bn;
    an = m[i];
    bn = m[i]-1;
    for (unsigned int _=0; _<iter; _++) {
      an = an*(1-bn/2);
      bn = (bn*bn)*((bn-3)/4);
    }
    m[i] = an;
  }
  CHECK_DIFF(m, msqrt);
TEST_DONE();

  memcpy(m, m0, sizeof(m0));
  he_ecd(&pt, m);
  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

TEST_DO("sqrt");
  he_sqrt(&ct_sqrt, &ct, &rlk, iter);
  he_show_ct_params(&ct_sqrt, "ct_sqrt");
  he_dec(&pt, &ct_sqrt, &sk);
  he_dcd(m, &pt);
  CHECK_DIFF(m, msqrt);
TEST_DONE();

  he_free_pt(&pt);
  he_free_ct(&ct);
  he_free_ct(&ct_sqrt);
  he_free_evk(&rlk);

TEST_END();
}

static void test_bootstrap()
{
TEST_BEGIN();

  he_bootstrapctx_init();
  printf("%s %u\n", __func__, __LINE__);

  if (!strcmp(key, "sk"))
    he_enc_sk(&ct, &pt, &sk);
  if (!strcmp(key, "pk"))
    he_enc_pk(&ct, &pt, &pk);

  printf("%s %u\n", __func__, __LINE__);
  he_bootstrapctx_exit();

TEST_END();
}

void set_params(int argc, char *argv[])
{
  if ((argc==1) || (argc==2&&strcmp(argv[1], "ecd"))) {
    fprintf(stderr, "usage: %s [ecd/enc/add/mul/conj/rot/gemv/inv] [sk/pk] "
      "--logn=num --logq=num --logp=num --iter=num\n", argv[0]);
    exit(1);
  }
  if (!strcmp(argv[1], "ecd"))
    return;
  strcpy(key, argv[2]);
  if ((!strcmp(argv[1], "enc"))
    ||(!strcmp(argv[1], "add"))
    ||(!strcmp(argv[1], "mul"))
    ||(!strcmp(argv[1], "conj"))
    ||(!strcmp(argv[1], "rot"))
    ||(!strcmp(argv[1], "gemv"))
    ||(!strcmp(argv[1], "sum"))
    ||(!strcmp(argv[1], "idx"))
    ||(!strcmp(argv[1], "nrm2"))) {
    logn     =  14; /*  12 */
    logq     = 438; /* 109 */
    slots    =  16; /*   4 */
    logDelta =  50; /*  30 */
  }
  if ((!strcmp(argv[1], "exp"))
    ||(!strcmp(argv[1], "log"))
    ||(!strcmp(argv[1], "sigmoid"))
    ||(!strcmp(argv[1], "inv"))
    ||(!strcmp(argv[1], "sqrt"))
    ||(!strcmp(argv[1], "cmp"))
    ||(!strcmp(argv[1], "rlsin"))) {
    logn = 14;
    logq  = 438;
    slots = 4;
    logDelta = 30;
    iter  = 5;
  }
  if (!strcmp(argv[1], "sqrt"))
    iter = 6;
  if (!strcmp(argv[1], "cmp")) {
    logn     =  15;
    logq     = 881;
    slots    =   4;
    logDelta =  30;
    iter     =   5;
    alpha    =   2;
  }
  unsigned int argidx=3; /* start index of parameter chosen */
  while (argv[argidx]) {
    if (!strncmp(argv[argidx], "--logn=", 7))
      logn=atoi(argv[argidx]+7), argidx++;
    else if (!strncmp(argv[argidx], "--logq=", 7))
      logq=atoi(argv[argidx]+7), argidx++;
    else if (!strncmp(argv[argidx], "--slots=", 8))
      slots=atoi(argv[argidx]+8), argidx++;
    else if (!strncmp(argv[argidx], "--logDelta=", 11))
      logDelta=atoi(argv[argidx]+11), argidx++;
    else if (!strncmp(argv[argidx], "--iter=", 7))
      iter=atoi(argv[argidx]+7), argidx++;
    else if (!strncmp(argv[argidx], "--alpha=", 8))
      alpha=atoi(argv[argidx]+8), argidx++;
    else if (!strncmp(argv[argidx], "--idx=", 6))
      idx=atoi(argv[argidx]+6), argidx++;
    else if (!strncmp(argv[argidx], "--start=", 8))
      start=atoi(argv[argidx]+8), argidx++;
    else if (!strncmp(argv[argidx], "--end=", 6))
      end=atoi(argv[argidx]+6), argidx++;
  }
}

int main(int argc, char *argv[])
{
  MPI q = mpi_set_ui(NULL, 1);
  set_params(argc, argv);
  mpi_lshift(q, q, logq);
  Delta = 1UL<<logDelta;
  hectx_init(logn, q, slots, Delta);
  he_show_ctx_params();
  /* init keys */
  he_alloc_pk(&pk);
  he_alloc_sk(&sk);
  /* main */
  if (!strcmp(argv[1], "ecd")){
    test_ecd();
    goto release;
  }
  he_keypair(&pk, &sk);
  if (!strcmp(argv[1], "enc"))
    test_enc ();
  if (!strcmp(argv[1], "add"))
    test_add ();
  if (!strcmp(argv[1], "mul"))
    test_mul ();
  if (!strcmp(argv[1], "conj"))
    test_conj();
  if (!strcmp(argv[1], "rot"))
    test_rot ();
  if (!strcmp(argv[1], "gemv"))
    test_gemv();
  if (!strcmp(argv[1], "sum"))
    test_sum ();
  if (!strcmp(argv[1], "idx"))
    test_idx();
  if (!strcmp(argv[1], "nrm2"))
    test_nrm2();
  if (!strcmp(argv[1], "inv"))
    test_inv ();
  if (!strcmp(argv[1], "exp"))
    test_exp ();
  if (!strcmp(argv[1], "sigmoid"))
    test_sigmoid();
  if (!strcmp(argv[1], "log"))
    test_log();
  if (!strcmp(argv[1], "cmp"))
    test_cmp();
  if (!strcmp(argv[1], "coeff2slot"))
    test_coeff2slot();
  if (!strcmp(argv[1], "rlsin"))
    test_rlsin();
  if (!strcmp(argv[1], "sqrt"))
    test_sqrt();
  if (!strcmp(argv[1], "bootstrap"))
    test_bootstrap();

release:
  /* release */
  mpi_release(q);
  he_free_sk(&sk);
  he_free_pk(&pk);
  hectx_exit();
  return 0;
}
