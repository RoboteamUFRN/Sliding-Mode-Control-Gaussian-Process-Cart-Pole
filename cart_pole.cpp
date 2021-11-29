#include <cmath>
#include <iostream>
#include <time.h>
#include <fstream>
#include <stdio.h>
#define _USE_MATH_DEFINES // for C++
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#define ARMA_USE_CXX11
#include <armadillo>
using namespace std;
using namespace arma;
#include "gp.h"

double m_car = 0.5;
double m_pen = 0.2;
double m_car_est = 0.85*m_car;
double m_pen_est = 0.85*m_pen;
double l = 0.2;
double g = 9.81;

double Maa(double);
double Muu(double);
double Maa_est(double);
double Muu_est(double);
double Mau(double);
double fa(double, double, double, double);
double fu(double, double);
double Mau_est(double);
double fa_est(double, double);
double fu_est(double);
double Maalinha(double);
double Muulinha(double);
double Maalinha_est(double);
double Muulinha_est(double);
double falinha(double, double, double, double, double);
double fulinha(double, double, double, double, double);
double falinha_est(double, double, double, double);
double fulinha_est(double, double, double, double);
double qapp(double, double, double, double, double, double);
double qupp(double, double, double, double, double, double);
double sat(double);
void rk4_2(double, double, double*, double*, double*, double*, double);


double edo(double, double, double, double);

int main()
{
	FILE *saida;
	saida = fopen("results.txt", "w");
	if (!saida)
	{
		perror("File opening failed");
		return EXIT_FAILURE;
	}
	TS = fopen("results_TS.txt", "w");
	if (!TS)
	{
		perror("File opening failed");
		return EXIT_FAILURE;
	}

	double t = 0.0;
	double tf = 39.98;
	double u, stf, atf = 0.0;
	double csr = 500.0;
	double cst = 1.0 / csr;
	double ssr = 1000.0;
	double sst = 1.0 / ssr;
	double asr = 50.0;
	double ast = 1.0 / asr;

	double thd, thpd, thppd, xd, xpd, xppd;
	double Ea, Eap, Eu, Eup, s, srp, Ms, fs, ganho, tau;
	double da, du, dalinha, dulinha;
	double x = 0.0;
	double xp = 0.0;
	double th = 6.0*M_PI/180.0;
	double thp = 0.0;
	double alpha_a = 0.02;
	double alpha_u = 1.0*l;
	double lambda_a = 0.05;
	double lambda_u = 2.5*l;
	double eta = 2.0;
	double delta = 0.5;
	double phi = 0.05;
	bool escolha = 1;
    double dr = 0.0;

	thd = 0.0;
	thpd = 0.0;
	thppd = 0.0;
	xd = 0.0;
	xpd = 0.0;
	xppd = 0.0;

    vec input, output, d_est, dr1;
    input.set_size(N_inputs);
    output.set_size(N_outputs);
    d_est.set_size(N_outputs);
    dr1.set_size(N_outputs);
    hyperpar = {0.2, 1.5, 0.01};//hiperparâmetros - {sigma_n, sigma_f, ell}
    double std_ds = 0.0;
    gpst = 100.0*cst;//period/(double (N_window));

    double int_u = 0.0;
    double int_du = 0.0;
    double int_t_s = 0.0;
    double u_old = 0.0;

	while (t < tf) {

            Ea = x - xd;
            Eap = xp - xpd;
            Eu = th - thd;
            Eup = thp - thpd;
            s = alpha_a*Eap + alpha_u*Eup + lambda_a*Ea + lambda_u*Eu;
            Ms = alpha_a*(1.0/Maalinha_est(th)) - alpha_u*(1.0/Muulinha_est(th))*Mau_est(th)*(1.0/Maa_est(th));
            fs = alpha_a*(1.0/Maalinha_est(th))*falinha_est(x, xp, th, thp) + alpha_u*(1.0/Muulinha_est(th))*fulinha_est(x, xp, th, thp);
            srp = - alpha_a*xppd - alpha_u*thppd + lambda_a*Eap + lambda_u*Eup;
            if(escolha == false){
                ganho = eta + delta;
                tau =  - (1.0/Ms) * (fs + srp + ganho*sat(s/phi) );
            }else{
                ganho = eta + 0.0*fabs(d_est(0)) + 2.0*sqrt(fabs(std_ds));
                tau =  - (1.0/Ms) * (fs + d_est(0) + srp + ganho*sat(s/phi) );
            }

            input[0] = s;

            da = Maa_est(x)*qapp(t, x, xp, th, thp, tau) + Mau_est(th)*qupp(t, x, xp, th, thp, tau) - fa_est(th, thp) - tau;
            du = Mau_est(th)*qapp(t, x, xp, th, thp, tau) + Muu_est(x)*qupp(t, x, xp, th, thp, tau) - fu_est(th);

            dalinha = da - Mau_est(th)*(1.0/Muu_est(x))*du;
            dulinha = du - Mau_est(th)*(1.0/Maa_est(x))*da;

            dr = alpha_a*(1.0/Maalinha_est(th))*dalinha + alpha_u*(1.0/Muulinha_est(th))*dulinha;
            output(0) = dr;

            stf = t + cst;
            while (t < stf) {
                rk4_2(t, sst, &x, &xp, &th, &thp, tau);
                t = t + sst;
            }

            Ea = x - xd;
            Eap = xp - xpd;
            Eu = th - thd;
            Eup = thp - thpd;
            s = alpha_a*Eap + alpha_u*Eup + lambda_a*Ea + lambda_u*Eu;

            input[0] = s;
            GPR(t, input, output, d_est, &std_ds);

            int_u += fabs(tau)*cst;
            int_du += fabs(tau-u_old);
            int_t_s += t*(alpha_a*fabs(Eap) + alpha_u*fabs(Eup) + lambda_a*fabs(Ea) + lambda_u*fabs(Eu))*cst;
            u_old = tau;

            if(t>atf){
                fprintf(saida, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", t, x, xd, th, thd, tau, s, dr, d_est(0), std_ds);
                atf = atf + ast;
            }
        }

        fprintf(saida, "#%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", hyperpar(0), hyperpar(1), hyperpar(2), gpst, int_t_s, int_u, int_du);
        printf("%d\n",N_real);

	fclose(saida);
	fclose(TS);

    return 0;
}

double Maa(double x){

    return (m_car+m_pen);

}
double Maa_est(double x){

    return (m_car_est+m_pen_est);

}
double Muu(double x){

    return (m_pen*l*l);

}
double Muu_est(double x){

    return (m_pen_est*l*l);

}
double Mau(double th){

    return (m_pen*l*cos(th));

}
double Mau_est(double th){

    return (m_pen_est*l*cos(th));

}
double fa(double t, double th, double thp, double xp){

    return (m_pen*l*thp*thp*sin(th) + coulomb(xp));

}
double fa_est(double th, double thp){

    return (m_pen_est*l*thp*thp*sin(th));

}
double fu(double t, double th){

    return m_pen*l*g*sin(th);

}
double fu_est(double th){

    return (m_pen_est*l*g*sin(th));

}
double Maalinha(double th){

    double r;

    r = Maa(th) - Mau(th)*(1.0/Muu(th))*Mau(th);

    return r;

}
double Maalinha_est(double th){

    double r;

    r = Maa_est(th) - Mau_est(th)*(1.0/Muu_est(th))*Mau_est(th);

    return r;

}
double Muulinha(double th){

    double r;

    r = Muu(th) - Mau(th)*(1.0/Maa(th))*Mau(th);

    return r;

}
double Muulinha_est(double th){

    double r;

    r = Muu_est(th) - Mau_est(th)*(1.0/Maa_est(th))*Mau_est(th);

    return r;

}
double falinha(double t, double x, double xp, double th, double thp){

    double r;

    r = fa(t, th, thp, xp) - Mau(th)*(1.0/Muu(x))*fu(t, th);

    return r;

}
double falinha_est(double x, double xp, double th, double thp){

    double r;

    r = fa_est(th, thp) - Mau_est(th)*(1.0/Muu_est(x))*fu_est(th);

    return r;

}
double fulinha(double t, double x, double xp, double th, double thp){

    double r;

    r = fu(t, th) - Mau(th)*(1.0/Maa(x))*fa(t, th, thp, xp);

    return r;

}
double fulinha_est(double x, double xp, double th, double thp){

    double r;

    r = fu_est(th) - Mau_est(th)*(1.0/Maa_est(x))*fa_est(th, thp);

    return r;

}
double qapp(double t, double x, double xp, double th, double thp, double tau){

    double r;

    r = (1.0/Maalinha(th))*(falinha(t, x, xp, th, thp) + tau);

    return r;

}
double qupp(double t, double x, double xp, double th, double thp, double tau){

    double r;

    r = (1.0/Muulinha(th))*(fulinha(t, x, xp, th, thp) - Mau(th)*(1.0/Maa(x))*tau);

    return r;

}
double sat(double x){

    if(x > 1.0){
        return 1.0;
    }else if(x < -1.0){
        return -1.0;
    }else{
        return x;
    }

}
void rk4_2(double t, double dt, double *x, double *xp, double *th, double *thp, double u){

    double k1p_x, k1v_x, k2p_x, k2v_x, k3p_x, k3v_x, k4p_x, k4v_x;
    double k1p_th, k1v_th, k2p_th, k2v_th, k3p_th, k3v_th, k4p_th, k4v_th;
    double X, Xp, TH, THp;

    X = *x;
    Xp = *xp;
    TH = *th;
    THp = *thp;

    k1p_x = dt*Xp;
    k1v_x = dt*qapp(t + dt/2.0, X, Xp, TH, THp, u);
    k1p_th = dt*THp;
    k1v_th = dt*qupp(t + dt/2.0, X, Xp, TH, THp, u);

    k2p_x = dt*(Xp + k1v_x/2.0);
    k2v_x = dt*qapp(t + dt/2.0, X + k1p_x/2.0, Xp + k1v_x/2.0, TH + k1p_th/2.0, THp + k1v_th/2.0, u);
    k2p_th = dt*(THp + k1v_th/2.0);
    k2v_th = dt*qupp(t + dt/2.0, X + k1p_x/2.0, Xp + k1v_x/2.0, TH + k1p_th/2.0, THp + k1v_th/2.0, u);

    k3p_x = dt*(Xp + k2v_x/2.0);
    k3v_x = dt*qapp(t + dt/2.0, X + k2p_x/2.0, Xp + k2v_x/2.0, TH + k2p_th/2.0, THp + k2v_th/2.0, u);
    k3p_th = dt*(THp + k2v_th/2.0);
    k3v_th = dt*qupp(t + dt/2.0, X + k2p_x/2.0, Xp + k2v_x/2.0, TH + k2p_th/2.0, THp + k2v_th/2.0, u);

    k4p_x = dt*(Xp + k3v_x);
    k4v_x = dt*qapp(t + dt, X + k3p_x, Xp + k3v_x, TH + k3p_th, THp + k3v_th, u);
    k4p_th = dt*(THp + k3v_th);
    k4v_th = dt*qupp(t + dt, X + k3p_x, Xp + k3v_x, TH + k3p_th, THp + k3v_th, u);

    X =  X  + (k1p_x + 2.0*k2p_x + 2.0*k3p_x + k4p_x)/6.0;
    Xp = Xp + (k1v_x + 2.0*k2v_x + 2.0*k3v_x + k4v_x)/6.0;
    TH =  TH  + (k1p_th + 2.0*k2p_th + 2.0*k3p_th + k4p_th)/6.0;
    THp = THp + (k1v_th + 2.0*k2v_th + 2.0*k3v_th + k4v_th)/6.0;

    *x = X;
    *xp = Xp;
    *th = TH;
    *thp = THp;

}


