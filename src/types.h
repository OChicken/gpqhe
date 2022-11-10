/*
 * Types definitions and conversions.
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

#ifndef TYPES_H
#define TYPES_H

#include "config.h"

#ifndef CONFIG_H
# error config.h must be included before types.h
#endif

#include <stddef.h>
#include <stdint.h>
#include <gcrypt.h>

/* The size of `uint64_t', as computed by sizeof. */
_Static_assert(sizeof(uint64_t) == 8,
  "Require sizeof(uint64_t) == 8");
#define SIZEOF_UINT64_T 8

#define BITS_PER_INT  32u
#define BITS_PER_LONG 64u
#define BITS_PER_LIMB SIZE_WIDTH
#define UINT64MAX_PER_DBLMAX 15

BEGIN_DECLS

typedef __uint128_t u128;
typedef struct gcry_mpi *MPI;

uint64_t mpi_to_u64 (MPI a);
int64_t  mpi_to_s64 (MPI a);
double mpi_to_double(MPI a);
void mpi_rdiv(MPI q, const MPI a, const MPI m);
void mpi_smod(MPI r, const MPI q, const MPI qh);

END_DECLS

#endif /* TYPES_H */
