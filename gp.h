
int N_window = 20;
int N_real = 0;
int N_inputs = 1;
int N_outputs = 1;
mat inputs_TS(N_inputs,N_window), outputs_TS(N_outputs,N_window), inputs_TS_aux, outputs_TS_aux;
vec kxX;
mat kXX_sI, ikXX_sI, kXX, ikXX;
vec hyperpar = {1.0, 1.0, 1.0};{sigma_n, sigma_f, ell} - default
double gptf = 0.0;
double gpst = 0.0;
int flag_TS = 0;

FILE *TS;

void COV()
{
	for (int i = 0; i<N_real; ++i) {
		for (int j = i; j<N_real; ++j) {
			kXX(i,j) = hyperpar(1)*hyperpar(1)*exp(-0.5*dot(inputs_TS_aux.col(i)-inputs_TS_aux.col(j),inputs_TS_aux.col(i)-inputs_TS_aux.col(j)) / (hyperpar(2)*hyperpar(2)));
			kXX(j,i) = kXX(i,j);
		}
	}
	kXX_sI = kXX + hyperpar(0)*hyperpar(0)*eye(N_real,N_real);
}


void COVV(arma::vec input)
{
	for (int i = 0; i<N_real; ++i) {
		kxX(i) = hyperpar(1)*hyperpar(1)*exp(-0.5*dot(inputs_TS_aux.col(i)-input,inputs_TS_aux.col(i)-input) / (hyperpar(2)*hyperpar(2)));
	}
}

double COVVxx(arma::vec input)
{
	double out;

	out = hyperpar(1)*hyperpar(1)*exp(-0.5*dot(input-input,input-input) / (hyperpar(2)*hyperpar(2)));

	return out;
}

void GPR(double t, arma::vec input, arma::vec output, arma::vec& d_est, double *std_ds){

    double Std_ds, gamma_tau = 0.0;
    vec ikXX_sI_kxX, alpha_tau;

    if(t >= gptf){
        if(N_real == 0){
            N_real++;
            inputs_TS_aux.set_size(N_inputs,N_real);
            outputs_TS_aux.set_size(N_outputs,N_real);
            kXX.set_size(N_real,N_real);
            ikXX.set_size(N_real,N_real);
            kXX_sI.set_size(N_real,N_real);
            ikXX_sI.set_size(N_real,N_real);
            kxX.set_size(N_real);
            alpha_tau.set_size(N_real);
            ikXX_sI_kxX.set_size(N_real);
            for(int i = 0; i < N_inputs; ++i){
                inputs_TS(i,N_real-1) = input(i);
            }
            for(int i = 0; i < N_outputs; ++i){
                outputs_TS(i,N_real-1) = output(i);
            }
            for(int i = 0; i < N_real; ++i){
                inputs_TS_aux.col(i) = inputs_TS.col(i);
                outputs_TS_aux.col(i) = outputs_TS.col(i);
            }
            COV();
            ikXX_sI = inv(kXX_sI);
            //ikXX = inv(kXX);
            fprintf(TS,"%.6f %.6f %.6f\n", t, input(0), output(0));
            gptf = t + gpst;
        }else{
            if(N_real < N_window && flag_TS == 0){
                N_real++;
                inputs_TS_aux.set_size(N_inputs,N_real);
                outputs_TS_aux.set_size(N_outputs,N_real);
                kXX.set_size(N_real,N_real);
                ikXX.set_size(N_real,N_real);
                kXX_sI.set_size(N_real,N_real);
                ikXX_sI.set_size(N_real,N_real);
                kxX.set_size(N_real);
                alpha_tau.set_size(N_real);
                ikXX_sI_kxX.set_size(N_real);
                for(int i = 0; i < N_inputs; ++i){
                    inputs_TS(i,N_real-1) = input(i);
                }
                for(int i = 0; i < N_outputs; ++i){
                    outputs_TS(i,N_real-1) = output(i);
                }
                //inputs_TS.col(N_real-1) = input;
                //outputs_TS.col(N_real-1) = output;
            }
            if(flag_TS == 1){
                COVV(input);
                alpha_tau = ikXX_sI*kxX;
                gamma_tau = COVVxx(input) - dot(kxX, alpha_tau);
                //printf("%f\n", gamma_tau);
                if(gamma_tau >= 0.0){
                    //printf("%f\n", gamma_tau);
                    for(int i = 0; i < N_real-1; ++i){
                        inputs_TS.col(i) = inputs_TS.col(i+1);
                        outputs_TS.col(i) = outputs_TS.col(i+1);
                    }
                    inputs_TS.col(N_real-1) = input;
                    outputs_TS.col(N_real-1) = output;
                }
            }
            if(N_real == N_window && flag_TS == 0){
                flag_TS = 1;
            }
            for(int i = 0; i < N_real; ++i){
                inputs_TS_aux.col(i) = inputs_TS.col(i);
                outputs_TS_aux.col(i) = outputs_TS.col(i);
            }
            COV();
            ikXX_sI = inv(kXX_sI);
            //ikXX = inv(kXX);
            fprintf(TS,"%.6f %.6f %.6f\n", t, input(0), output(0));
            gptf = t + gpst;
        }
    }

    if(N_real > 0){
        COVV(input);
        d_est = kxX.t()*ikXX_sI*outputs_TS_aux.t();
        ikXX_sI_kxX = ikXX_sI*kxX;
        Std_ds = COVVxx(input) - dot(kxX,ikXX_sI_kxX);
    }else{
        d_est.fill(0.0);
        Std_ds = 0.0;
    }
    //if(Std_ds < 0.0){Std_ds = 0.0;}
    *std_ds = Std_ds;
}

double sinal(double x){
    if(x > 0.0){
        return 1.0;
    }else if(x < 0.0){
        return -1.0;
    }else{
        return 0.0;
    }
}

double Heavside(double x){

    double r;

    r = (1.0 + sinal(x))/2.0;

    return r;

}

double coulomb(double xp){

    double mu = 0.4;
    double Fa;

    Fa = 1.0*mu*sinal(xp);

    return Fa;

}

