/**
 *	@file		mathematics.hpp
 *	@brief		Includes linear algebra, basic statstics, differential equation and integrals, Fourier transforms, and other mathematics headers.
 *	@author		seonho.oh@gmail.com
 *	@date		2013-07-01
 *	@version	1.0
 *
 *	@section	LICENSE
 *
 *		Copyright (c) 2013-2015, Seonho Oh
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

//!	@defgroup	math	Mathematics
//!	@brief		Includes headers implement linear algebra, basic statistics, differentiation and integrals, Fourier transforms, and other mathematics.
//!	@{
//!		@defgroup	elem	Elementary Math
//!		@brief		Trigonometry, exponentials and logathms, complex values, rounding, remainders, discrete math, coordinate system conversion.
//!		@{
//!			@defgroup	arit	Arithmetic
//!			@brief		Addition, subtraction, multiplication, left and right division, power.
//
//!			@defgroup	poly	Polynomials
//!			@brief		Curve fitting, roots, partial fraction expansions.
//!		@}
//
//!		@defgroup	linalg	Linear Algebra
//!		@brief		Matrix analysis, linear equations, eigenvalues, singular values, logarithms, exponentials, factorization.
//!		@{
//!			@defgroup	matanal	Matrix Analysis
//!			@brief		Norm, rank, determinant, condition.
//!		@}
//
//!	@defgroup	staran	Statistics and Random Numbers
//!	@brief		Descriptive statistics, random number generation.
//!	@{
//!		@defgroup	stat	Descriptive Statistics
//!		@brief		Range, central tendency, standard deviation, variance, correlation.
//
//!		@defgroup	rand	Random Number Generation
//!		@brief		Seeds, distributions, algorithms.
//!	@}
//
//!	@defgroup	interp	Interpolation
//
//!	@defgroup	optim	Optimization
//
//!	@defgroup	numintdiff	Numerical Integration and Differentiation
//!	@brief		Quadratures, double and triple integrals, and multidimensional derivatives.
//
//!	@defgroup	foufilt	Fourier Analysis and Fitlering
//!	@brief		Fourier transforms, convolution, digital filtering.
//
//!	@defgroup	sparse	Sparse Matrices
//
//!	@defgroup	comp	Computational Geometry
//
//!	@}

#include "arithmetic.hpp"	// basic arithmetic
#include "polynomials.hpp"	// polynomial
#include "matanal.hpp"		// matrix analysis
#include "statistics.hpp"	// statistics
#include "random.hpp"		// random number generation
#include "filtering.hpp"	// digital filtering

#include "numint.hpp"	// numerical integration
#include "numdiff.hpp"	// numerical differentiation