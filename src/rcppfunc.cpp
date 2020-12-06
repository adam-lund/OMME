//// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "auxfunc.h"

using namespace std;
using namespace arma;

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// maximin implementation ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//[[Rcpp::export]]
Rcpp::List admm(arma::mat dims, int usex, arma::mat B, arma::mat Phi1, arma::mat Phi2, arma::mat Phi3,
               Rcpp::NumericVector resp,
               std::string penalty,
               double kappa,
               int usekappa,
               arma::vec lambda,
               int nlambda,
               int makelamb,
               double lambdaminratio,
               arma::mat penaltyfactor,
               double reltol,
               int maxiter,
               int steps,
               std::string alg,
               int bound){

Rcpp::List output;
Rcpp::NumericVector vecresp(resp);
Rcpp::IntegerVector respDim = vecresp.attr("dim");
const arma::cube Y(vecresp.begin(), respDim[0], respDim[1], respDim[2], false);


int Tf=2,stopsparse,
  endmodelno = nlambda,
    n1 = dims(0,0), n2 = dims(1,0), n3 = dims(2,0),
  Nog = Y.n_slices, ng = n1 * n2 * n3,
    p1 = dims(0,1), p2 = dims(1,1), p3 = dims(2,1),
  p = p1 * p2 * p3, n=n1*n2*n3  ;

double  ascad=0.1, epsiloncor=0.2, c=10000.0,alphamax, delta, deltamax,l, in1, lambdamax,
  in2, in3,  tauk, gamk, alphahat, betahat, sparsity=0;

arma::vec df(nlambda),lambdamaxs(Nog),
          eig1, eig2, eig3,
          Iter(nlambda),  Pen(maxiter)      ;

arma::mat  Beta(p1, p2 * p3), Betatilde(p1, p2 * p3),  sumB(p1,p2*p3),

  Z(p1, p2 * p3), U(p1, p2 * p3),
           Zold(p1, p2 * p3),
          L(p1, p2 * p3), Betas(p, nlambda),
          Betak0(p1, p2 * p3),Zk0(p1, p2 * p3), Lk0(p1, p2 * p3), Lhatk0(p1, p2 * p3),
          Delta(maxiter, nlambda),  dpen(p1, p2 * p3),
          Gamma(p1, p2 * p3),
          Phi1tPhi1, Phi2tPhi2, Phi3tPhi3, PhitPhiBeta, PhitPhiX,
          wGamma(p1, p2 * p3),
Lhat(p1, p2 * p3),
deltaLhat, deltaL,
deltaHhat,
deltaGhat;

////fill variables
Betas.fill(42);
Iter.fill(0);

////precompute ///// compute group ols estimates!!!!
if(makelamb == 1){
sumB.fill(0);
}
for(int j = 0; j < Nog; j++){
arma::mat tmp;

if(usex == 1){
tmp = RHmat(Phi3.t(), RHmat(Phi2.t(), RHmat(Phi1.t(), Y.slice(j), n2, n3), n3, p1), p1, p2);
B.col(j) = vectorise(tmp);
}else{
tmp = reshape(B.col(j), p1, p2*p3);
}

////make lambda sequence
if(makelamb == 1){//fix?!?!? what is this used for and how?!!!???!

arma::mat   absgradzeroall = abs(tmp) %      penaltyfactor;
arma::mat absgradzeropencoef = absgradzeroall % (penaltyfactor > 0);
arma::mat penaltyfactorpencoef = (penaltyfactor == 0) * 1 + penaltyfactor;
lambdamaxs(j) = as_scalar(max(max(absgradzeropencoef / penaltyfactorpencoef)));
sumB = sumB + tmp;

}

}

if(makelamb == 1){//lambmax is the soft maximin lamb max...
  lambdamax = max(max(abs(sumB / Nog)));
}

if(makelamb == 1){

  if(usekappa == 1){//go from 0 towards lambmax linearly
    double M = 0;
    double difflamb = log(2) / pow(nlambda,4); //exp
    //double difflamb =  lambdamax/ nlambda; //lin
    lambda(0) = 0;
    for(int i = 1; i < (nlambda + 0) ; i++){
      lambda(i) = lambdamax * (exp(pow(i,4) * difflamb) - 1); //exponential
      //lambda(i) =   lambda(i - 1) + difflamb; //linear
    }

//   double M = 0;
// //double difflamb = log(2) / nlambda; //abs(M - m) / (nlambda - 1);
// double difflamb =  lambdamax/ nlambda; //abs(M - m) / (nlambda - 1);
// //double l = 0;
// lambda(0) = 0;
// for(int i = 1; i < (nlambda + 0) ; i++){
//
// //lambda(i) = lambdamax * (exp(i * difflamb) - 1);
// lambda(i) =   lambda(i - 1) + difflamb;
// //l = l - difflamb;
//
// }

}else{//go lambmax to lambmin exponentially
  double lambdamax = 2 * max(lambdamaxs) / Nog;
  double m = log(lambdaminratio);
  double M = 0;
  double difflamb = abs(M - m) / (nlambda - 1);
  double l = 0;
  for(int i = 0; i < (nlambda - 1) ; i++){
    lambda(i) = lambdamax * exp(i * difflamb);
    l = l - difflamb;
}
}
}else{std::sort(lambda.begin(), lambda.end()//, std::greater<int>()
                  );}//fix!!! sort decreasing ot not??!!!


///// /////initilize how!?!?!
////set up projection problem
arma::mat AE(Nog, 1), AI(Nog, Nog), ce(1, 1), ci(Nog, 1);
AE.ones();
Eigen::MatrixXd eigen_AE = eigen_cast(AE);
ce.fill(-1);
Eigen::VectorXd eigen_ce = eigen_cast(ce);
AI.eye();
Eigen::MatrixXd eigen_AI = eigen_cast(AI);
ci.zeros();
Eigen::VectorXd eigen_ci = eigen_cast(ci);

////this computation can be halfed by symmetry but maybe thats done on by arma already??
arma::mat Q = B.t() * B;
Eigen::MatrixXd eigen_Q = eigen_cast(Q);

//// fixed step size COULD WE USE THE LIP CONSTANT???!!!////precompute what to do with no phi?!?!?!?!
// Phi1tPhi1 = Phi1.t() * Phi1;
// Phi2tPhi2 = Phi2.t() * Phi2;
// Phi3tPhi3 = Phi3.t() * Phi3;
// eig1 = arma::eig_sym(Phi1tPhi1);
// eig2 = arma::eig_sym(Phi2tPhi2);
// eig3 = arma::eig_sym(Phi3tPhi3);
// alphamax = as_scalar(max(kron(eig1, kron(eig2 , eig3))));

l =  4  / pow(ng, 2) ; //?!?!?!?!?!?! //upper bound on Lipschitz constant
//delta =  1.9 / l; //stepsize scaled up by  nu
deltamax = 1.99 / l; //maximum theoretically allowed stepsize

double ABSTOL   = 1e-7; //FIX!?!?!
double RELTOL   = 1e-7; //FIX!?!?!
delta = 1;//delta?!?!?!fixed stepsize!!!!!!!
double rho = 1/delta; //FIX!?!?!
tauk = 0.1; //FIX!?!?!
gamk = 1.0; //FIX!?!?!

Beta =  reshape(B.col(1), p1, p2*p3);
Z= reshape(B.col(2), p1, p2*p3);
U.fill(0);
L= Beta + Z;

Betak0 = Beta;
Zk0 = Z;
Lk0 = L;
Lhatk0 = Lhat;

arma::vec hr_norm(maxiter), hs_norm(maxiter), heps_dual(maxiter), heps_pri(maxiter);

///////////start lambda loop
for (int m = 0; m < nlambda; m++){
Gamma = penaltyfactor * lambda(m);

//start MSA loop
for (int s = 0; s < steps; s++){

if(s == 0){//fix!!!!! division by zero for non lasso penalty!!!

if(penalty != "lasso"){wGamma = Gamma / lambda(m);}else{wGamma = Gamma;}

}else{///fix!!!!! division by zero for non lasso penalty!!!

if(penalty == "scad"){//can we use this ???!?!? scad is multi lasso and each lasso is ok...

arma::mat absBeta = abs(Beta);
  arma::mat pospart = ((ascad * Gamma - absBeta) + (ascad * Gamma - absBeta)) / 2;
  arma::mat dpen = sign(Beta) % Gamma % ((absBeta <= Gamma) + pospart / (ascad - 1) % (absBeta > Gamma));
wGamma = abs(dpen) % Gamma / lambda(m) % (Beta != 0) + lambda(m) * (Beta == 0);

}

}

/////////////////ADMM algorithm from ...boyd ... figuerido
for (int k = 0; k < maxiter; k++){

if(alg == "admm"){//boyd
//x-update, prox operator for f
Beta = prox(Z - U, wGamma , delta);

//z-update, projection onto to convex hull of columns in B
Zold = Z;
Z = proj(B, Beta + U, eigen_Q, eigen_AE, eigen_ce, eigen_AI, eigen_ci, p1, p2, p3);

//u-update
U = U + Beta - Z;
}
if(alg == "aradmm"){  //L/tauk=-U!!!

//update
  Beta = prox(Z - U, wGamma , 1 / tauk); // tauk isnt setpsize but multiplier!!
  Betatilde = gamk * Beta + (1 - gamk) * Z;
  Zold = Z;
  Z = proj(B, Beta + U, eigen_Q, eigen_AE, eigen_ce, eigen_AI, eigen_ci, p1, p2, p3);
  L = L + tauk * (Z - Betatilde);
  U = -L / tauk; //=-L/tauk+Betatilde-Z;

//step size calculation
if((k % Tf) == 1){
deltaL = L - Lk0;
Lhat = L + tauk * (-Beta + Zold);
deltaLhat = Lhat - Lhatk0;
deltaHhat = Beta - Betak0;
deltaGhat = -(Z - Zk0);

//could be its own function???
in1 = accu(square(deltaLhat));
in2 = accu(square(deltaHhat));
in3 = accu(deltaHhat % deltaLhat);
double alphahatsd = in1 / in3;
double alphahatmg = in3 / in2;
double alphacor = in3 / (sqrt(in1) * sqrt(in2));
if(alphahatmg > alphahatsd / 2){
alphahat = alphahatmg;
}else{
alphahat = alphahatsd - alphahatmg / 2;
}

//beta correct?!?
in1 = accu(square(deltaL));
in2 = accu(square(deltaGhat));
in3 = accu(deltaGhat % deltaL);
double betahatsd = in1 / in3;
double betahatmg = in3 / in2;
double betacor = in3 / (sqrt(in1) * sqrt(in2));
if(betahatmg > betahatsd / 2){
betahat = betahatmg;
}else{
betahat = betahatsd - betahatmg / 2;
}

//update steps
if((alphacor > epsiloncor) && (betacor > epsiloncor)){
tauk = sqrt(alphahat * betahat);
gamk = 1 + 2 * sqrt(alphahat * betahat) / (alphahat + betahat);
}
else if((alphacor > epsiloncor) && (betacor <= epsiloncor)){
tauk = alphahat;
gamk = 1.9;
}
else if((alphacor <= epsiloncor) && (betacor > epsiloncor)){
tauk = betahat;
gamk = 1.1;
}
else{gamk=1.5;}

//bound stpes
if(bound == 1){
tauk = min(tauk, (1 + c / pow(k,2)) * tauk);
gamk = min(gamk, 1 + c / pow(k,2));
}

//store k0 values
Betak0 = Beta;
Zk0 = Z;
Lk0 = L;
Lhatk0 = Lhat;

}

}
Iter(m) = k + 1;

// diagnostics, reporting, termination checks IS ng CORRECT???????
hr_norm(k) = accu(square(Beta - Z));
hs_norm(k) = accu(square(-rho * (Z - Zold)));
heps_pri(k) = sqrt(ng) * ABSTOL + RELTOL * max(accu(square(Beta)), accu(square(-Z)));
heps_dual(k) = sqrt(ng) * ABSTOL + RELTOL * accu(square(rho* U));
if((//k>3 &&
   (hr_norm(k) < heps_pri(k)) && (hs_norm(k) < heps_dual(k))) || (k == maxiter - 1)){//k>3!??!?!!?!?
  arma::mat stBetavec = vectorise(st(Beta, wGamma));
  Betas.col(m) = stBetavec; //is wGAMMA correct????????
    sparsity = accu(stBetavec == 0) ;
  df(m) = p - sparsity;
    break;
}//

///EHAT TO DO IF K ==MAXITER??????!?!?!?!

}//end admm loop

//Stop msa loop if maximum number of backtracking steps or maxiter is reached
//if(Stopbt == 1 || Stopmaxiter == 1){
//
// endmodelno = j;
// break;
//
// }

}//end MSA loop
//Stop lambda loop if maximum number of backtracking steps or maxiter is reached
// endmodelno = m;
//
// if(kappa <  sparsity / (p * 1.0) //|| Stopmaxiter == 1
//      ){
//  stopsparse=1;
//  // m=nlambda;
//   break;
//
// }

endmodelno = m + 1;

if((usekappa == 1 & kappa <  sparsity / (p * 1.0))){
  stopsparse=1;
  break;
}


}//end lambda loop

// Stops(0) = Stopconv;
// Stops(1) = Stopmaxiter;


output = Rcpp::List::create(Rcpp::Named("Beta") = Betas,
                             Rcpp::Named("df") = df,
                            Rcpp::Named("B") = B,
                            Rcpp::Named("stopsparse") = stopsparse,
                            Rcpp::Named("Z") = kappa,
                            Rcpp::Named("U") = sparsity,
                            Rcpp::Named("Iter") = Iter,
                             Rcpp::Named("endmodelno") = endmodelno,
                             Rcpp::Named("lambda") = lambda
                            // Rcpp::Named("L") = L,
                           // Rcpp::Named("Stops") = Stops
                              );

return output;

}


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// magging implementation /////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//[[Rcpp::export]]
Rcpp::List maglas(arma::mat dims, int usex,
                  arma::mat B, arma::mat Phi1, arma::mat Phi2, arma::mat Phi3,
                    Rcpp::NumericVector resp,
                    std::string penalty,
                    double kappa,
                    int usekappa,
                    arma::vec lambda,
                    int nlambda,
                    int makelamb,
                    double lambdaminratio,
                    arma::mat penaltyfactor,
                    int steps){

  Rcpp::List output;
  Rcpp::NumericVector vecresp(resp);
  Rcpp::IntegerVector respDim = vecresp.attr("dim");
  const arma::cube Y(vecresp.begin(), respDim[0], respDim[1], respDim[2], false);


  int Tf=2,stopsparse = 0,
    endmodelno = nlambda,
    n1 = dims(0,0), n2 = dims(1,0), n3 = dims(2,0),
    Nog = Y.n_slices, ng = n1 * n2 * n3,
    p1 = dims(0,1), p2 = dims(1,1), p3 = dims(2,1),
    p = p1 * p2 * p3, n=n1*n2*n3  ;

  double  ascad=0.1, sparsity= 0;

  arma::vec df(nlambda),lambdamaxs(Nog)     ;

  arma::mat  stB, Q, Betas(p, nlambda), Penaltyfactor(p1* p2 * p3, Nog), //stBs(Nog * p1* p2 * p3, nlambda),
  dpen(p1, p2 * p3),
  Gamma(p1, p2 * p3),          wGamma(p1, p2 * p3), posdefscale;


  ////fill variables
  Betas.fill(42);
  posdefscale.eye(Nog,Nog);
  posdefscale = posdefscale * 0.000001;
  ////precompute ///// compute group ols estimates!!!!
  for(int j = 0; j < Nog; j++){
    Penaltyfactor.col(j)= vectorise(penaltyfactor);
    arma::mat tmp;
    if(usex == 1){
      tmp = RHmat(Phi3.t(), RHmat(Phi2.t(), RHmat(Phi1.t(), Y.slice(j), n2, n3), n3, p1), p1, p2);
      B.col(j) = vectorise(tmp);
    }else{
      tmp = reshape(B.col(j), p1, p2*p3);
    }
    if(makelamb == 1){
      arma::mat   absgradzeroall = abs(tmp) % penaltyfactor;
      arma::mat absgradzeropencoef = absgradzeroall % (penaltyfactor > 0);
      arma::mat penaltyfactorpencoef = (penaltyfactor == 0) * 1 + penaltyfactor;
      lambdamaxs(j) = as_scalar(max(max(absgradzeropencoef / penaltyfactorpencoef)));
    }

  }

  ////set up quad  problem
  arma::mat AE(Nog, 1), AI(Nog, Nog), ce(1, 1), ci(Nog, 1);
  AE.ones();
  Eigen::MatrixXd eigen_AE = eigen_cast(AE);
  ce.fill(-1);
  Eigen::VectorXd eigen_ce = eigen_cast(ce);
  AI.eye();
  Eigen::MatrixXd eigen_AI = eigen_cast(AI);
  ci.zeros();
  Eigen::VectorXd eigen_ci = eigen_cast(ci);
  Eigen::MatrixXd eigen_Q;

  ////make lambda sequence
  if(makelamb == 1){
  if(usekappa == 1){//go from 0 towards lambmax linearly/expo
    double lambdamax = max(lambdamaxs);
    double M = 0;
    double difflamb = log(2) / pow(nlambda,4); //exp
    //double difflamb =  lambdamax/ nlambda; //lin
    //double l = 0;
    lambda(0) = 0;
    for(int i = 1; i < (nlambda + 0) ; i++){

      lambda(i) = lambdamax * (exp(pow(i,4) * difflamb) - 1); //exponential
      //lambda(i) =   lambda(i - 1) + difflamb; //linear
      //l = l - difflamb;
    }

  }else{//go lambmax to lambmin exponentially as normal
    arma::mat Ze = zeros<mat>(n1, n2 * n3);
    double lambdamax = max(lambdamaxs);
    double m = log(lambdaminratio);
    double M = 0;
    double difflamb = abs(M - m) / (nlambda - 1);
    double l = 0;

    for(int i = 0; i < nlambda ; i++){

      lambda(i) = lambdamax * exp(l);
      l = l - difflamb;
  }
  }
  }else{std::sort(lambda.begin(), lambda.end(), std::greater<int>());}


  ///////////start lambda loop
  for (int m = 0; m < nlambda; m++){
    Gamma = Penaltyfactor * lambda(m);

    //start MSA loop
    for (int s = 0; s < steps; s++){

      if(s == 0){

        if(penalty != "lasso"){wGamma = Gamma / lambda(m);}else{wGamma = Gamma;}

      }else{

        if(penalty == "scad"){//can we use this ???!?!? scad is multi lasso and each lasso is ok...

//          HOW DOES THIS WORK?! ITERATION HHOW?!?!

          // arma::mat absBeta = abs(Beta);
          // arma::mat pospart = ((ascad * Gamma - absBeta) + (ascad * Gamma - absBeta)) / 2;
          // arma::mat dpen = sign(Beta) % Gamma % ((absBeta <= Gamma) + pospart / (ascad - 1) % (absBeta > Gamma));
          // wGamma = abs(dpen) % Gamma / lambda(m) % (Beta != 0) + lambda(m) * (Beta == 0);

        }

      }
      ////this computation can be halvfed by symmetry but maybe thats done on by arma already??
      stB = st(B, wGamma);
     // stBs.col(m) = vectorise(stB);
      Q = stB.t() * stB + posdefscale;
      eigen_Q = eigen_cast(Q);

      // Betas.col(m) = magging(stB, eigen_Q,eigen_AE, eigen_ce, eigen_AI, eigen_ci);
      arma::mat tmp = magging(stB, eigen_Q, eigen_AE, eigen_ce, eigen_AI, eigen_ci);
      sparsity = accu(tmp == 0);
      Betas.col(m)=tmp;
        df(m)= p - sparsity;

    }//end MSA loop
     endmodelno = m + 1;

    if((usekappa == 1 & kappa <  sparsity / (p * 1.0))){
       stopsparse=1;
      break;
    }


  }//end lambda loop

  output = Rcpp::List::create(Rcpp::Named("Beta") = Betas,
                              Rcpp::Named("stopsparse") = stopsparse,
                              Rcpp::Named("B") = B,
                              Rcpp::Named("df") = df,
                              Rcpp::Named("endmodelno") = endmodelno,
                              Rcpp::Named("lambda") = lambda
  );

  return output;

}
