/**
 *	@file		statistics.hpp
 *	@brief		An implemenation of statistics operations
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

	/**
	 *	@brief	2-D correlation coefficient
	 *	@param A
	 *	@param B
	 *	@return	the correlation coefficient between @c A and @c B, where @c A and @c B are matrices or vectors of the same size.
	 *			@c corr2 computes the correlation coefficient using
	 *			\f[
	 *				r = \frac{ \sum_{m}\sum_{n}(A_{mn}-\bar{A}) (B_{mn}-\bar{B}) }{\sqrt{ \left( \sum_{m}\sum_{n}(A_{mn}-\bar{A})^2 \right) \left( \sum_{m}\sum_{n}(B_{mn}-\bar{B})^2 \right)}}
	 *			\f]
	 *			where
	 *			\f$ \bar{A}\f$=mean2(A), and \f$\bar{B}\f$=mean2(B).
	 *	@see	http://www.mathworks.co.kr/kr/help/images/ref/corr2.html
	 */
	template <typename mat_type, typename prec_type>
	inline prec_type corr2(const mat_type& A, const mat_type& B)
	{
		typedef Mat<prec_type> tmat;
		tmat A1, B1;
		A1 = conv_to<tmat>::from(A);
		B1 = conv_to<tmat>::from(B);
		A1 -= mean2(A1);
		B1 -= mean2(B1);
		return accu(A1 % B1) / sqrt(accu(square(A1)) * accu(square(B1)));
	}

	/**
	 *	@brief	Distance metric
	 */
	enum distance_type : uword {
		euclidean,		///< Euclidean distance.
		seuclidean,		///< Standarized Eucliean distance.
		cityblock,		///< City block metric.
		minkowski,		///< Minkowski distance. The default exponent is 2.
		chebychev,		///< Chebychev distance (maximum coordinate difference).
		mahalanobis,	///< Mahalanobis distance, using the sample covariance of X.
		cosine,			///< One minus the cosine of the included angle between points(treated as vectors).
		correlation,	///< One minus the sample correlation between points (treatedas sequences of values).
		spearman,		///< One minus the sample Spearman's rank correlation betweenobservations (treated as sequences of values).
		hamming,		///< Hamming distance, which is the percentage of coordinatesthat differ.
		jaccard,		///< One minus the Jaccard coefficient, which is the percentageof nonzero coordinates that differ.
		custom
	};

	typedef double (*pdist_func)(const arma::subview_row<double>&, const arma::subview_row<double>&);

	/// Euclidean distance for pdist
	double pdist_euclidean(const arma::subview_row<double>& a, const arma::subview_row<double>& b)
	{
		return sqrt(sum(square(b - a)));
	}

	/**
	 *	@brief	Pairwise distance between pairs of objects.<br>
	 *			Computes the distance between pairs of objects in \f$m\f$-by-\f$n\f$ data matrix \f$X\f$.
	 *			Rows of \f$X\f$ correspond to observations, and columns correspond to variables.
	 *			Output is the row vector of length \f$frac{m(m - 1)}{2}\f$, corresponding to pairs of observations in \f$X\f$.
	 *			The distances are arranged in the order \f$(2, 1), (3, 1), \cdots, (m, 1), (3, 2), \cdots, (m, 2), \cdots, (m, m - 1)\f$.
	 *			Output is commonly used as a dissimilarity matrix in clustering or multidimensional scailing.
	 *	@return	Pairwise distance.
	 */
	vec pdist(const mat& X, distance_type type = euclidean, pdist_func func_ptr = nullptr)
	{
		const uword m = X.n_rows;
		vec Y(m * (m - 1) / 2);
		double* ptr = Y.memptr();

		switch (type) {
		case euclidean:
			func_ptr = pdist_euclidean;
			break;
		case custom:
			// stub
			break;
		default:
			func_ptr = &pdist_euclidean;
		}

		uword k = 0;
		for (uword i = 0 ; i < m ; i++)
			for (uword j = i + 1 ; j < m ; j++)
				ptr[k++] = func_ptr(X.row(i), X.row(j)); //sqrt(sum(square(X.row(j) - X.row(i))));

		return Y;
	}
}