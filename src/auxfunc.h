/*
    A number of functions utilized by rcppfunc.cpp.

    Intended for use with R.
    Copyright (C) 2020 Adam Lund

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <math.h>
using namespace std;
//using namespace arma;

///new
#include <RcppEigen.h>

#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Dense>

#include "eiquadprog.h"

using namespace Rcpp;
using namespace Eigen;

////////////////////////////////// Auxiliary functions
//////////////////// Direct RH-transform of a flat 3d array (matrix) M by a matrix X
arma::mat RHmat(arma::mat const& X, arma::mat const& M,int col, int sli){

int rowx = X.n_rows;

////matrix multiply
arma::mat XM = X * M;

////make matrix into rotated (!) cube (on matrix form)
arma::mat Mnew(col, sli * rowx);
for (int s = 0; s < sli; s++) {

for (int c = 0; c < col; c++) {

for (int r = 0; r < rowx; r++) {

Mnew(c, s + r * sli) = XM(r, c + s * col);

}

}

}

return Mnew;

}


//////////////////// Sum of squares function
double sum_square(arma::mat const& x){return accu(x % x);}

//////////////////// soft threshold operator on matrix
arma::mat st(arma::mat const& zv, arma::mat const& gam){

return (zv >= gam) % (zv - gam) + (zv <= -gam) % (zv + gam);

}



// [[Rcpp::export]]
arma::mat prox(arma::mat const& z, arma::mat const& gam, double del){
    arma::mat t, tmp;
    tmp.zeros(z.n_rows,z.n_cols);
    t = gam*del;
    tmp = tmp + (z < -gam - t) % ((z - t) / (1 + 2 * del)); //coordinat wise div
    tmp = tmp + (z >= -gam - t) % (z <= -gam) % (-gam);
    tmp = tmp + (abs(z) < gam) % z;
    tmp = tmp + (z <= gam + t) % (z>= gam) % gam;
    tmp = tmp + (z > gam + t) % ((z + t) / (1 + 2 * del)); //coordinat wise div

    return tmp;

}

#include <RcppArmadillo.h>
#include <RcppEigen.h>

// // [[Rcpp::depends(RcppEigen)]]
//
// // [[Rcpp::depends(RcppArmadillo)]]
//
// // [[Rcpp::export]]
// Eigen::MatrixXd example_cast_eigen(arma::mat arma_A) {
//
//   Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
//                                                         arma_A.n_rows,
//                                                         arma_A.n_cols);
//
//   return eigen_B;
// }

//// [[Rcpp::export]]
arma::mat arma_cast(Eigen::MatrixXd eigen_A) {
arma::mat arma_B = arma::mat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(),
                               true,   // changed from false to true.
                               false);
  // arma::mat arma_B = arma::mat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(),
  //                              false, false);

  return arma_B;
}

// // [[Rcpp::export]]
Eigen::MatrixXd eigen_cast(arma::mat arma_A) {

Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
                                                          arma_A.n_rows,
                                                          arma_A.n_cols);

    return eigen_B;
}

arma::mat proj(arma::mat const& B,
               arma::mat const& z
                 ,  Eigen::MatrixXd & Q,
                 const Eigen::MatrixXd & AE,
                 const Eigen::VectorXd & ce,
                 const Eigen::MatrixXd & AI,
                 const Eigen::VectorXd & ci,
               int p1, int p2, int p3
){

arma::vec  zvec=vectorise(z);
Eigen::VectorXd q = eigen_cast(-B.t() * zvec);//?????why the minus?? THIS COULD BE EXPENSIVE!!! G*p
Eigen::VectorXd w(B.n_cols);
solve_quadprog(Q, q, AE, ce, AI, ci, w);

return reshape(B * arma_cast(w), p1, p2 * p3); ///THIS COULD BE EXPENSIVE!!! p*G

}

// //  [[Rcpp::export]]
// arma::mat projr(arma::mat const& B,
//                arma::mat const& z
//                  ,  Eigen::MatrixXd & Q,
//                  int p1, int p2, int p3
// ){
//
//   // Eigen::MatrixXd & G,
//   // Eigen::VectorXd & g0,
//   // const Eigen::MatrixXd & CE,
//   // const Eigen::VectorXd & ce0,
//   // const Eigen::MatrixXd & CI,
//   // const Eigen::VectorXd & ci0) {
//   // const int n = G.rows();
//   //     const int p = ce0.size();
//   //     const int m = ci0.size();
//   //    double f = solve_quadprog(G, g0,  CE, ce0,  CI, ci0, x);
//
//
//   arma::mat AE(B.n_cols,1), AI(B.n_cols,B.n_cols), ce(1,1), ci(B.n_cols,1);
//   AE.ones();
//   Eigen::MatrixXd eigen_AE = eigen_cast(AE);
//   ce.fill(-1);
//   Eigen::VectorXd eigen_ce = eigen_cast(ce);
//   AI.eye();
//   Eigen::MatrixXd eigen_AI = eigen_cast(AI);
//   ci.zeros();
//   Eigen::VectorXd eigen_ci = eigen_cast(ci);
//
//   arma::mat out(p1*p2*p3,1);
//
//   arma::vec  zvec=vectorise(z);
//   Eigen::VectorXd q = eigen_cast(-B.t() * zvec);//?????why the minus?? THIS COULD BE EXPENSIVE!!! G*p
//   Eigen::VectorXd w(B.n_cols);
//   //Eigen::VectorXd eigen_w = eigen_cast(w);
//
//   double f = solve_quadprog(Q, q, eigen_AE, eigen_ce, eigen_AI, eigen_ci, w);
//
//   //double f = solve_quadprog(Q, q, AE, ce, AI, ci, w);
//
//   out = B * arma_cast(w); ///THIS COULD BE EXPENSIVE!!! p*G
//   return reshape(out, p1, p2 * p3); //!??!!?!?CORRET DIM?
//
//   //out =  arma_cast(w); ///THIS COULD BE EXPENSIVE!!! p*G
//   //return out; //!??!!?!?CORRET DIM?
//
//   // Rcpp::List output;
//   //
//   // output = Rcpp::List::create(Rcpp::Named("Q") = Q,
//   //                             Rcpp::Named("q") = q,
//   //                             Rcpp::Named("AI") = AI,
//   //                             Rcpp::Named("AE") = AE,
//   //                             Rcpp::Named("ce") = ce,
//   //                             Rcpp::Named("ci") = ci,
//   //                        //     Rcpp::Named("w") = w,
//   //                             Rcpp::Named("eigen_Q") = eigen_Q,
//   //                             Rcpp::Named("eigen_q") = eigen_q,
//   //                             Rcpp::Named("eigen_AI") = eigen_AI,
//   //                             Rcpp::Named("eigen_AE") = eigen_AE,
//   //                             Rcpp::Named("eigen_ce") = eigen_ce,
//   //                             Rcpp::Named("eigen_ci") = eigen_ci,
//   //                             Rcpp::Named("eigen_w") = eigen_w);
//   //
//   //return  output;
//
//
// }



// double ST1a(double z,double gam){
//     double sparse=0;
//     if(z>0 && gam<fabs(z)) return(z-gam);
//
//     if(z<0 && gam<fabs(z)) return(z+gam);
//     if(gam>=fabs(z)) return(sparse);
//     else return(0);
// }

// [[Rcpp::export]]
arma::mat  magging(arma::mat const& B,
                  Eigen::MatrixXd & Q,
                  const Eigen::MatrixXd & AE,
                  const Eigen::VectorXd & ce,
                  const Eigen::MatrixXd & AI,
                  const Eigen::VectorXd & ci){

  arma::mat q(B.n_cols, 1);
  q.fill(0);
  Eigen::VectorXd qeigen = eigen_cast(q);
  Eigen::VectorXd w(B.n_cols);
  double f=solve_quadprog(Q, qeigen, AE, ce, AI, ci, w);
//arma::mat F(1,1);
//F(0,0)=f;
  return //F;//
  B * arma_cast(w); ///THIS COULD BE EXPENSIVE!!! p*G

}

//////////////////// The weighted (gam = penaltyfactor * lambda) l1-penalty function
double l1penalty(arma::mat const& gam, arma::mat const& zv){return accu(gam % abs(zv));}

//////////////////// The weighted (gam = penaltyfactor * lambda) scad-penalty function
double scadpenalty(arma::mat const& gam, double a, arma::mat const& zv){

arma::mat absbeta = abs(zv);

return accu(gam % absbeta % (absbeta <= gam) - (pow(zv, 2) - 2 * a * gam % absbeta + pow(gam, 2)) / (2 * (a - 1)) % (gam < absbeta && absbeta <= a * gam) + (a + 1) * pow(gam, 2) / 2 % (absbeta > a * gam));

}
