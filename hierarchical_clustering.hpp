/**
 *	@file		hierarchical_clustering.hpp
 *	@brief		Produce nested sets of clusters
 *	@author		seonho.oh@gmail.com
 *	@date		2013-07-01
 *	@version	1.0
 *
 *	@section	LICENSE
 *
 *		Copyright (c) 2007-2013, Seonho Oh
 *		All rights reserved. 
 * 
 *		Redistribution and use in source and binary forms, with or without  
 *		modification, are permitted provided that the following conditions are  
 *		met: 
 * 
 *		    * Redistributions of source code must retain the above copyright  
 *		    notice, this list of conditions and the following disclaimer. 
 *		    * Redistributions in binary form must reproduce the above copyright  
 *		    notice, this list of conditions and the following disclaimer in the  
 *		    documentation and/or other materials provided with the distribution. 
 *		    * Neither the name of the <ORGANIZATION> nor the names of its  
 *		    contributors may be used to endorse or promote products derived from  
 *		    this software without specific prior written permission. 
 * 
 *		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS  
 *		IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  
 *		TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A  
 *		PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER  
 *		OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  
 *		EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,  
 *		PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR  
 *		PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  
 *		LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  
 *		NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS  
 *		SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 */

#pragma once

#include <armadillo>

namespace arma_ext
{
	using namespace arma;

#ifndef DOXYGEN
	/**
	 *	@brief	Cut the tree at a specified point
	 */
	uvec checkcut(const mat& X, double cutoff, const vec& crit)
	{
		// See which nodes are below the cutoff, disconnect thoese that aren't
		uword n = X.n_rows + 1;
		uvec conn = (crit <= cutoff); // these are still connected
		
		// We may still disconnect a node unless all non-leaf children are
		// below the cutoff, and grand-children, and so on
		uvec todo = conn % ((X.col(0) > n) + (X.col(1) > n));

		while (any(todo)) {
			uvec rows = find(todo);

			// See if each child is done, or if it requires disconnecting its parent
			umat cdone = ones<umat>(rows.n_elem, 2);

			for (uword j = 0 ; j < 2 ; j++) { // 0: left child, 1: right child
				vec crows = vec(X.col(j)).elem(rows);
				uvec t = (crows > n);
				if (any(t)) {
					uvec ti = find(t);

					uvec child = conv_to<uvec>::from(crows(ti) - n);
					
					// cdone(t,j) = ~todo(child);
					// conn(rows(t)) = conn(rows(t)) & conn(child);
					for (uword k = 0 ; k < ti.n_elem ; k++) {
						uword tval = ti(k);
						uword childval = child(k) - 1;	// 0-based indexing
						cdone(tval, j) = todo(childval) ? 0 : 1;
						conn(rows(tval)) = conn(rows(tval)) & conn(childval);
					}
				}
			}

			// update todo list
			todo(rows(find(cdone.col(0) % cdone.col(1)))).fill(0);
		}

		return conn;
	}

	/**
	 *	@brief	Assign cluster number
	 */
	uvec labeltree(const mat& X, uvec conn)
	{
		uword n = X.n_rows;
		uword nleaves = n + 1;
		uvec T = ones<uvec>(n + 1);

		// Each cut potentially yeild as additional cluster
		uvec todo = ones<uvec>(n);

		// Define cluster numbers for each side of each non-leaf node
		umat clustlist = reshape(arma_ext::sequence<uvec>(1, 2 * n), n, 2);

		// Propagate cluster numbers down the tree
		while (any(todo)) {
			// Work on rows that are now split but not yet processed
			// rows = find(todo & ~conn);
			uvec rows = find(todo % (ones<uvec>(n) - conn));
			if (rows.empty()) break;

			for (uword j = 0 ; j < 2 ; j++) {	// 0: left, 1: right
				uvec children = conv_to<uvec>::from(X.col(j)).elem(rows);

				// Assign cluster number to child leaf node
				uvec leaf = (children <= nleaves);
				if (any(leaf)) {
					uvec leafi = find(leaf);
					std::for_each(leafi.begin(), leafi.end(), [&](uword index) {
						T(children(index) - 1) = clustlist(rows(index), j);
					});
				}

				// Also assign it to both children of any joined child non-leaf nodes
				uvec joint = ones<uvec>(n) - leaf;	// ~leaf
				uvec jointi = find(joint);
				joint(jointi) = conn(children(jointi) - nleaves - 1);

				if (any(joint)) {
					std::for_each(jointi.begin(), jointi.end(), [&](uword index) {
						uword clustnum = clustlist(rows(index), j);
						uword childnum = children(index) - nleaves - 1;
						clustlist.row(childnum - 1).fill(clustnum);
						conn(childnum) = 0;
					});
				}
			}

			// Mark these rows as done
			todo(rows).fill(0);
		}

		return unique(T);
	}

	/**
	 *	@note	This function taken from linkagemex.cpp, and partially adopted.
	 *			This function could have copyright problem.
	 *	@copyright 2003-2006 The MathWorks, Inc.
	 */
	template <typename mat_type>
	mat linkagemex(const mat_type& X)
	{
		#define ISNAN(a) (a != a)

		enum method_types
		{single,complete,average,weighted,centroid,median,ward} method_key;

		typedef int mwSize;
		typedef double TEMPL;

		static TEMPL  inf;
		mwSize        m,m2,m2m3,m2m1,n,i,j,bn,bc,bp,p1,p2,q,q1,q2,h,k,l,g;
		mwSize        nk,nl,ng,nkpnl,sT,N;
		mwSize        *obp,*scl,*K,*L;
		TEMPL         *y,*yi,*s,*b1,*b2,*T;
		TEMPL         t1,t2,t3,rnk,rnl;
		int           uses_scl = false,  no_squared_input = true;

		/* get the method */
		method_key = single;

		/* get the dimensions of inputs */
		n = (mwSize)X.size();	/* number of pairwise distances --> n */
		m = (mwSize)std::ceil(std::sqrt(2.0 * n));	/* size of distance matrix --> m = (1 + sqrt(1+8*n))/2 */

		/*  create a pointer to the input pairwise distances */
		yi = const_cast<double*>(X.colptr(0));

		/* set space to copy the input */
		y = (TEMPL *) malloc(n * sizeof(TEMPL));

		/* copy input and compute Y^2 if necessary.  lots of books use 0.5*Y^2
		* for ward's, but the 1/2 makes no difference */
		if (no_squared_input) memcpy(y,yi,n * sizeof(TEMPL));
		else /* then it is ward's, centroid, or median */
			for (i=0; i<n; i++) y[i] = yi[i] * yi[i];

		/* calculate some other constants */
		bn   = m-1;                        /* number of branches     --> bn */
		m2   = m * 2;                      /* 2*m */
		m2m3 = m2 - 3;                     /* 2*m - 3 */
		m2m1 = m2 - 1;                     /* 2*m - 1 */

		inf = arma::datum::inf;

		/*  allocate space for the output matrix  */
		mat out(bn, 3);
		b1 = out.colptr(0);	/*leftmost  column */
		b2 = b1 + bn;		/*center    column */
		s = b2 + bn;		/*rightmost column */

		/* find the best value for N (size of the temporal vector of  */
		/* minimums) depending on the problem size */
		if      (m>1023) N = 512;
		else if (m>511)  N = 256;
		else if (m>255)  N = 128;
		else if (m>127)  N = 64;
		else if (m>63)   N = 32;
		else             N = 16;

		if (method_key == single) N = N >> 2;

		/* set space for the vector of minimums (and indexes) */
		T = (TEMPL *)malloc(N * sizeof(TEMPL));
		K = (mwSize *)malloc(N * sizeof(mwSize));
		L = (mwSize *)malloc(N * sizeof(mwSize));

		/* set space for the obs-branch pointers  */
		obp = (mwSize *) malloc(m * sizeof(mwSize));

		switch (method_key) {
		case average:
		case centroid:
		case ward:
			uses_scl = true;
			/* set space for the size of clusters vector */
			scl = (mwSize *) malloc(m * sizeof(mwSize));
			/* initialize obp and scl */
			for (i=0; i<m; obp[i]=i, scl[i++]=1);
			break;
		default: /*all other cases */
			/* only initialize obp */
			for (i=0; i<m; i++) obp[i]=i;
		} /* switch (method_key) */

		sT = 0;  t3 = inf;

		for (bc=0,bp=m;bc<bn;bc++,bp++){
			/* *** MAIN LOOP ***
			bc is a "branch counter" --> bc = [ 0:bn-1]
			bp is a "branch pointer" --> bp = [ m:m+bc-1 ], it is used to point
			branches in the output since the values [0:m-1]+1 are reserved for
			leaves.
			*/

			/* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /*
			find the "k","l" indices of the minimum distance "t1" in the remaining
			half matrix, the new computed distances to the new cluster will be placed
			in the row/col "l", then the leftmost column in the matrix of pairwise
			distances will be moved to the row/col "k", so the whole matrix of
			distances is smaller at every step */

			/*  OLD METHOD: search for the minimun in the whole "y" at every branch
			iteration
			t1 = inf;
			p1 = ((m2m1 - bc) * bc) >> 1; /* finds where the remaining matrix starts
			for (j=bc; j<m; j++) {
			for (i=j+1; i<m; i++) {
			t2 = y[p1++];
			if (t2<t1) { k=j, l=i, t1=t2;}
			}
			}
			*/

			/*  NEW METHOD: Keeps a sorted vector "T" with the N minimum distances,
			at every branch iteration we only pick the first entry. Now the whole
			"y" is not searched at every step, we will search it again only when
			all the entries in "T" have been used or invalidated. However, we need
			to keep track of invalid distances already sorted in "y", and also
			update the index vectors "K" and "L" with permutations occurred in the
			half matrix "y"
			*/

			/* cuts "T" so it does not contain any distance greater than any of the
			new distances computed when joined the last clusters ("t3" contains
			the minimum new distance computed in the last iteration). */
			for (h=0;((T[h]<t3) && (h<sT));h++);
			sT = h; t3 = inf;
			/* ONLY when "T" is empty it searches again "y" for the N minimum
			distances  */
			if (sT==0) {
				for (h=0; h<N; T[h++]=inf);
				p1 = ((m2m1 - bc) * bc) >> 1; /* finds where the matrix starts */
				for (j=bc; j<m; j++) {
					for (i=j+1; i<m; i++) {
						t2 = y[p1++];
						/*  this would be needed to solve NaN bug in MSVC*/
						/*  if (!mxIsNaN(t2)) { */
						if (t2 <= T[N-1]) {
							for (h=N-1; ((t2 <= T[h-1]) && (h>0)); h--) {
								T[h]=T[h-1];
								K[h]=K[h-1];
								L[h]=L[h-1];
							} /* for (h=N-1 ... */
							T[h] = t2;
							K[h] = j;
							L[h] = i;
							sT++;
						} /* if (t2<T[N-1]) */
						/*}*/
					} /*  for (i= ... */
				} /* for (j= ... */
				if (sT>N) sT=N;
			} /* if (sT<1) */

			/* if sT==0 but bc<bn then the remaining distances in "T" must be
			NaN's ! we break the loop, but still need to fill the remaining
			output rows with linkage info and NaN distances
			*/
			if (sT==0) break;


			/* the first entry in the ordered vector of distances "T" is the one
			that will be used for this branch, "k" and "l" are its indexes */
			k=K[0]; l=L[0]; t1=T[0];

			/* some housekeeping over "T" to inactivate all the other minimum
			distances which also have a "k" or "l" index, and then also take
			care of those indexes of the distances which are in the leftmost
			column */
			for (h=0,i=1;i<sT;i++) {
				/* test if the other entries of "T" belong to the branch "k" or "l"
				if it is true, do not move them in to the updated "T" because
				these distances will be recomputed after merging the clusters */
				if ( (k!=K[i]) && (l!=L[i]) && (l!=K[i]) && (k!=L[i]) ) {
					T[h]=T[i];
					K[h]=K[i];
					L[h]=L[i];
					/* test if the preserved distances in "T" belong to the
					leftmost column (to be permutated), if it is true find out
					the value of the new indices for such entry */
					if (bc==K[h]) {
						if (k>L[h]) {
							K[h] = L[h];
							L[h] = k;
						} /* if (k> ...*/
						else K[h] = k;
					} /* if (bc== ... */
					h++;
				} /* if k!= ... */
			} /* for (h=0 ... */
			sT=h; /* the new size of "T" after the shifting */

			/* Update output for this branch, puts smaller pointers always in the
			leftmost column */
			if (obp[k]<obp[l]) {
				*b1++ = (TEMPL) (obp[k]+1); /* +1 since Matlab ptrs start at 1 */
				*b2++ = (TEMPL) (obp[l]+1);
			} else {
				*b1++ = (TEMPL) (obp[l]+1);
				*b2++ = (TEMPL) (obp[k]+1);
			}
			*s++ = (no_squared_input) ? t1 : sqrt(t1);

			/* Updates obs-branch pointers "obp" */
			obp[k] = obp[bc];        /* new cluster branch ptr */
			obp[l] = bp;             /* leftmost column cluster branch ptr */

			/* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /*
			Merges two observations/clusters ("k" and "l") by re-calculating new
			distances for every remaining observation/cluster and place the
			information in the row/col "l" */

			/*

			example:  bc=2  k=5  l=8   bn=11   m=12

			0
			1    N                             Pairwise
			2    N   N                         Distance
			3    N   N   Y                     Half Matrix
			4    N   N   Y   Y
			5    N   N  p1*  *   *
			6    N   N   Y   Y   Y   +
			7    N   N   Y   Y   Y   +   Y
			8    N   N  p2*  *   *   []  +   +
			9    N   N   Y   Y   Y   o   Y   Y   o
			10   N   N   Y   Y   Y   o   Y   Y   o   Y
			11   N   N   Y   Y   Y   o   Y   Y   o   Y   Y

			0   1   2   3   4   5   6   7   8   9   10   11


			p1 is the initial pointer for the kth row-col
			p2 is the initial pointer for the lth row-col
			*  are the samples touched in the first loop
			+  are the samples touched in the second loop
			o  are the samples touched in the third loop
			N  is the part of the whole half matrix which is no longer used
			Y  are all the other samples (not touched)

			*/

			/* computing some limit constants to set up the 3-loops to
			transverse Y */
			q1 = bn - k - 1;
			q2 = bn - l - 1;

			/* initial pointers to the "k" and  "l" entries in the remaining half
			matrix */
			p1 = (((m2m1 - bc) * bc) >> 1) + k - bc - 1;
			p2 = p1 - k + l;

			if (uses_scl) {
				/* Get the cluster cardinalities  */
				nk     = scl[k];
				nl     = scl[l];
				nkpnl  = nk + nl;

				/* Updates cluster cardinality "scl" */
				scl[k] = scl[bc];        /* letfmost column cluster cardinality */
				scl[l] = nkpnl;          /* new cluster cardinality */

			} /* if (uses_scl) */

			/* some other values that we want to compute outside the loops */
			switch (method_key) {
			case centroid:
				t1 = t1 * ((TEMPL) nk * (TEMPL) nl) / ((TEMPL) nkpnl * (TEMPL) nkpnl);
			case average:
				/* Computes weighting ratios */
				rnk = (TEMPL) nk / (TEMPL) nkpnl;
				rnl = (TEMPL) nl / (TEMPL) nkpnl;
				break;
			case median:
				t1 = t1/4;
			} /* switch (method_key) */

			switch (method_key) {
			case average:
				for (q=bn-bc-1; q>q1; q--) {
					t2 = y[p1] * rnk + y[p2] * rnl;
					if (t2 < t3) t3 = t2 ;
					y[p2] = t2;
					p1 = p1 + q;
					p2 = p2 + q;
				}
				p1++;
				p2 = p2 + q;
				for (q=q1-1;  q>q2; q--) {
					t2 = y[p1] * rnk + y[p2] * rnl;
					if (t2 < t3) t3 = t2 ;
					y[p2] = t2;
					p1++;
					p2 = p2 + q;
				}
				p1++;
				p2++;
				for (q=q2+1; q>0; q--) {
					t2 = y[p1] * rnk + y[p2] * rnl;
					if (t2 < t3) t3 = t2 ;
					y[p2] = t2;
					p1++;
					p2++;
				}
				break; /* case average */

			case single:
				for (q=bn-bc-1; q>q1; q--) {
					if (y[p1] < y[p2]) y[p2] = y[p1];
					else if (ISNAN(y[p2])) y[p2] = y[p1];
					if (y[p2] < t3)    t3 = y[p2];
					p1 = p1 + q;
					p2 = p2 + q;
				}
				p1++;
				p2 = p2 + q;
				for (q=q1-1;  q>q2; q--) {
					if (y[p1] < y[p2]) y[p2] = y[p1];
					else if (ISNAN(y[p2])) y[p2] = y[p1];
					if (y[p2] < t3)    t3 = y[p2];
					p1++;
					p2 = p2 + q;
				}
				p1++;
				p2++;
				for (q=q2+1; q>0; q--) {
					if (y[p1] < y[p2]) y[p2] = y[p1];
					else if (ISNAN(y[p2])) y[p2] = y[p1];
					if (y[p2] < t3)    t3 = y[p2];
					p1++;
					p2++;
				}
				break; /* case simple */

			case complete:
				for (q=bn-bc-1; q>q1; q--) {
					if (y[p1] > y[p2]) y[p2] = y[p1];
					else if (ISNAN(y[p2])) y[p2] = y[p1];
					if (y[p2] < t3)    t3 = y[p2];
					p1 = p1 + q;
					p2 = p2 + q;
				}
				p1++;
				p2 = p2 + q;
				for (q=q1-1;  q>q2; q--) {
					if (y[p1] > y[p2]) y[p2] = y[p1];
					else if (ISNAN(y[p2])) y[p2] = y[p1];
					if (y[p2] < t3)    t3 = y[p2];
					p1++;
					p2 = p2 + q;
				}
				p1++;
				p2++;
				for (q=q2+1; q>0; q--) {
					if (y[p1] > y[p2]) y[p2] = y[p1];
					else if (ISNAN(y[p2])) y[p2] = y[p1];
					if (y[p2] < t3)    t3 = y[p2];
					p1++;
					p2++;
				}
				break; /* case complete */

			case weighted:
				for (q=bn-bc-1; q>q1; q--) {
					t2 = (y[p1] + y[p2])/2;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1 = p1 + q;
					p2 = p2 + q;
				}
				p1++;
				p2 = p2 + q;
				for (q=q1-1;  q>q2; q--) {
					t2 = (y[p1] + y[p2])/2;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1++;
					p2 = p2 + q;
				}
				p1++;
				p2++;
				for (q=q2+1; q>0; q--) {
					t2 = (y[p1] + y[p2])/2;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1++;
					p2++;
				}
				break; /* case weighted */

			case centroid:
				for (q=bn-bc-1; q>q1; q--) {
					t2 = y[p1] * rnk + y[p2] * rnl - t1;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1 = p1 + q;
					p2 = p2 + q;
				}
				p1++;
				p2 = p2 + q;
				for (q=q1-1;  q>q2; q--) {
					t2 = y[p1] * rnk + y[p2] * rnl - t1;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1++;
					p2 = p2 + q;
				}
				p1++;
				p2++;
				for (q=q2+1; q>0; q--) {
					t2 = y[p1] * rnk + y[p2] * rnl - t1;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1++;
					p2++;
				}
				break; /* case centroid */

			case median:
				for (q=bn-bc-1; q>q1; q--) {
					t2 = (y[p1] + y[p2])/2 - t1;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1 = p1 + q;
					p2 = p2 + q;
				}
				p1++;
				p2 = p2 + q;
				for (q=q1-1;  q>q2; q--) {
					t2 = (y[p1] + y[p2])/2 - t1;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1++;
					p2 = p2 + q;
				}
				p1++;
				p2++;
				for (q=q2+1; q>0; q--) {
					t2 = (y[p1] + y[p2])/2 - t1;
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1++;
					p2++;
				}
				break; /* case median */

			case ward:
				for (q=bn-bc-1,g=bc; q>q1; q--) {
					ng = scl[g++];
					t2 = (y[p1]*(nk+ng) + y[p2]*(nl+ng) - t1*ng) / (nkpnl+ng);
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1 = p1 + q;
					p2 = p2 + q;
				}
				g++;
				p1++;
				p2 = p2 + q;
				for (q=q1-1;  q>q2; q--) {
					ng = scl[g++];
					t2 = (y[p1]*(nk+ng) + y[p2]*(nl+ng) - t1*ng) / (nkpnl+ng);
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1++;
					p2 = p2 + q;
				}
				g++;
				p1++;
				p2++;
				for (q=q2+1; q>0; q--) {
					ng = scl[g++];
					t2 = (y[p1]*(nk+ng) + y[p2]*(nl+ng) - t1*ng) / (nkpnl+ng);
					if (t2<t3) t3=t2;
					y[p2] = t2;
					p1++;
					p2++;
				}
				break; /* case ward */

			} /* switch (method_key) */

			/* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /* /*
			moves the leftmost column "bc" to row/col "k" */
			if (k!=bc) {
				q1 = bn - k;

				p1 = (((m2m3 - bc) * bc) >> 1) + k - 1;
				p2 = p1 - k + bc + 1;

				for (q=bn-bc-1; q>q1; q--) {
					p1 = p1 + q;
					y[p1] = y[p2++];
				}
				p1 = p1 + q + 1;
				p2++;
				for ( ; q>0; q--) {
					y[p1++] = y[p2++];
				}
			} /*if (k!=bc) */
		} /*for (bc=0,bp=m;bc<bn;bc++,bp++) */

		/* loop to fill with NaN's in case the main loop ended prematurely */
		for (;bc<bn;bc++,bp++) {
			k=bc; l=bc+1;
			if (obp[k]<obp[l]) {
				*b1++ = (TEMPL) (obp[k]+1);
				*b2++ = (TEMPL) (obp[l]+1);
			} else {
				*b1++ = (TEMPL) (obp[l]+1);
				*b2++ = (TEMPL) (obp[k]+1);
			}
			obp[l] = bp;
			*s++ = arma::datum::nan;
		}

		free(y);	/* destroy the copy of pairwise distances */

		if (uses_scl) free(scl);

		free(obp);
		free(L);
		free(K);
		free(T);

		return out;
	}
#endif

	/**
	 *	@brief	Agglomerative hierarchical cluster tree.
	 *	@param X The ream matrix of observation or the vector of pairwise distance of the observations.
	 *	@return A matrix that encodes a tree of hierarchical cluster of the rows of the real matrix X or the vector of pairwise distances of matrix.<br>
	 *			It is \f$(m - 1)\f$-by-\f$3\f$ matrix, where \f$ m \f$ is the number of observations in the original data.
	 *	@see	http://www.mathworks.co.kr/kr/help/stats/linkage.html
	 *	@note	This function is preliminary; it is not yet fully optimized.
	 *			This is an simplified implementtion of linkagemex function
	 */
	template <typename mat_type>
	mat linkage(const mat_type& X)
	{
		return linkagemex(X);
	}

	/**
	 *	@brief	Construct clusters from the agglomerative hierarchical cluster tree
	 *	@param Z The agglomerative hierarchical cluster tree, as generated by #linkage function.
	 *	@param c A threshold for cutting Z into clusters.
	 *	@return	The cluster indices for each of observations.
	 *	@see	http://www.mathworks.co.kr/kr/help/stats/cluster.html
	 *	@note	This function is preliminary; it is not yet fully optimized.
	 */
	uvec cluster(const mat& Z, double c)
	{
		uword m = Z.n_rows + 1;
		// distance cutoff criterion for forming clusters
		vec crit = Z.col(2);	// distance criterion

		uvec conn = checkcut(Z, c, crit);
		return labeltree(Z, conn);
	}
}