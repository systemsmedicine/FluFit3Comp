#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand.h>

#include "ranNumbers.h"

#define THS_MAX 256

#define FLAG \
        fprintf(stderr, "Flag in %s:%d\n", __FILE__, __LINE__);\

// Dormand-Prince coefficients
#define A21 0.2
#define A31 0.075
#define A32 0.225
#define A41 (44.0/45.0)
#define A42 (-56.0/15.0)
#define A43 (32.0/9.0)
#define A51 (19372.0/6561.0)
#define A52 (-25360/2187.0)
#define A53 (64448.0/6561.0)
#define A54 (-212.0/729.0)
#define A61 (9017.0/3168.0)
#define A62 (-355.0/33.0)
#define A63 (46732.0/5247.0)
#define A64 (49.0/176.0)
#define A65 (-5103.0/18656.0)
#define A71 (35.0/384.0)
#define A73 (500.0/1113.0)
#define A74 (125.0/192.0)
#define A75 (-2187.0/6784.0)
#define A76 (11.0/84.0)

/*=-=-=-=-=-=-=-=-=-=-=-=-=- STRUCTURES =-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

typedef struct 
{
	float U1;
	float I1;
	float R1;
	float V1;
	float F1;

	float U2;
	float I2;
	float R2;
	float V2;
	float F2;
	float T2;

	float U3;
	float I3;
	float R3;
	float V3;
	float F3;
	float T3;
} 
comp;

typedef struct 
{
	float time;
	float V1;
	float V2;
	float V3;
} 
viralData;

typedef struct
{
	float minTime;
	float maxTime;
	float minValue;
	float maxValue;
}
window;

typedef struct 
{
	float U10;
	float I10;
	float R10;
	float V10;
	float F10;

	float U20;
	float I20;
	float R20;
	float V20;
	float F20;
	float T20;

	float U30;
	float I30;
	float R30;
	float V30;
	float F30;
	float T30;

	float bet1;
	float xi1;
	float chi1;
	float del1;
	float rho1;
	float sig1;
	float alp1;
	float phi1;

	float bet2;
	float xi2;
	float chi2;
	float psi2;
	float del2;
	float rho2;
	float sig2;
	float alp2;
	float phi2;
	float eta2;
	float kap2;
	float ups2;

	float bet3;
	float xi3;
	float chi3;
	float psi3;
	float del3;
	float rho3;
	float sig3;
	float alp3;
	float phi3;
	float eta3;
	float kap3;
	float ups3;

	float gam12;
	float gam21;
	float gam23;
	float gam32;
	float zet23;
	float zet32;

	int m;

	float t0;
	float tN;
	float dt;

	int D;
	int Np;
	int nData;
	int qnData2;
	int qnData3;
	int qFlag;
	int sizeSample;
} 
param;

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=- FUNCTIONS =-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

long nextPow2(long x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

//-------------------------------------------------------------------------

__device__ void model(int idx, param pars, float *pop, comp Y, comp *dotY)
{
	int ii = 1;
	float bet1 = pop[idx + ii];
	ii++;
	float xi1  = pop[idx + ii];
	ii++;
	float chi1 = pop[idx + ii];
	ii++;
	float del1 = pop[idx + ii];
	ii++;
	float rho1 = pop[idx + ii];
	ii++;
	float sig1 = pop[idx + ii];
	ii++;
	float alp1 = pop[idx + ii];
	ii++;
	float phi1 = pop[idx + ii];
	ii++;

	float bet2 = pop[idx + ii];
	ii++;
	float xi2  = pop[idx + ii];
	ii++;
	float chi2 = pop[idx + ii];
	ii++;
	float psi2 = pop[idx + ii];
	ii++;
	float del2 = pop[idx + ii];
	ii++;
	float rho2 = pop[idx + ii];
	ii++;
	float sig2 = pop[idx + ii];
	ii++;
	float alp2 = pop[idx + ii];
	ii++;
	float phi2 = pop[idx + ii];
	ii++;
	float eta2 = pop[idx + ii];
	ii++;
	float kap2 = pop[idx + ii];
	ii++;
	float ups2 = pop[idx + ii];
	ii++;

	float bet3 = pop[idx + ii];
	ii++;
	float xi3  = pop[idx + ii];
	ii++;
	float chi3 = pop[idx + ii];
	ii++;
	float psi3 = pop[idx + ii];
	ii++;
	float del3 = pop[idx + ii];
	ii++;
	float rho3 = pop[idx + ii];
	ii++;
	float sig3 = pop[idx + ii];
	ii++;
	float alp3 = pop[idx + ii];
	ii++;
	float phi3 = pop[idx + ii];
	ii++;
	float eta3 = pop[idx + ii];
	ii++;
	float kap3 = pop[idx + ii];
	ii++;
	float ups3 = pop[idx + ii];
	ii++;

	float gam12 = pop[idx + ii];
	ii++;
	float gam21 = pop[idx + ii];
	ii++;
	float gam23 = pop[idx + ii];
	ii++;
	float gam32 = pop[idx + ii];
	ii++;
	float zet23 = pop[idx + ii];
	ii++;
	float zet32 = pop[idx + ii];

	float m = (float) pars.m;
	float T20 = pars.T20;
	float T30 = pars.T30;

	dotY->U1 = -bet1*Y.U1*Y.V1 - xi1*Y.U1*Y.I1 + chi1*Y.R1;
	dotY->I1 = bet1*Y.U1*Y.V1 - del1*Y.I1;
	dotY->R1 = xi1*Y.U1*Y.I1 - chi1*Y.R1;
	dotY->V1 = rho1*Y.I1 - sig1*Y.V1 - gam12*Y.V1 + gam21*Y.V2;
	dotY->F1 = alp1*Y.I1 - phi1*Y.F1;

	dotY->U2 = -bet2*Y.U2*Y.V2 - xi2*Y.U2*Y.I2 + chi2*Y.R2;
	dotY->I2 = bet2*Y.U2*Y.V2 - del2*Y.I2 - psi2*Y.I2*Y.T2;
	dotY->R2 = xi2*Y.U2*Y.I2 - chi2*Y.R2;
	dotY->V2 = rho2*Y.I2 - sig2*Y.V2
		- gam21*Y.V2 + gam12*Y.V1 - gam23*Y.V2 + gam32*Y.V3;
	dotY->F2 = alp2*Y.I2 - phi2*Y.F2;
	dotY->T2 = eta2*Y.T2*(pow(Y.V2,m)/(pow(Y.V2,m) + pow(kap2,m)))
		- ups2*Y.T2 + ups2*T20 - zet23*Y.T2 + zet32*Y.T3;

	dotY->U3 = -bet3*Y.U3*Y.V3 - xi3*Y.U3*Y.I3 + chi3*Y.R3;
	dotY->I3 = bet3*Y.U3*Y.V3 - del3*Y.I3 - psi3*Y.I3*Y.T3;
	dotY->R3 = xi3*Y.U3*Y.I3 - chi3*Y.R3;
	dotY->V3 = rho3*Y.I3 - sig3*Y.V3 - gam32*Y.V3 + gam23*Y.V2;
	dotY->F3 = alp3*Y.I3 - phi3*Y.F3;
	dotY->T3 = eta3*Y.T3*(pow(Y.V3,m)/(pow(Y.V3,m) + pow(kap3,m)))
		- ups3*Y.T3 + ups3*T30 - zet32*Y.T3 + zet23*Y.T2;

	return;
}

//-------------------------------------------------------------------------

__device__ void deriv_step(int idx, param pars, float *pop, comp *Y)
{
	float h = pars.dt;
    comp Yold, Ytemp, k1, k2, k3, k4, k5, k6;

	// Old Y values
	Yold.U1 = Y->U1;
	Yold.I1 = Y->I1;
	Yold.R1 = Y->R1;
	Yold.V1 = Y->V1;
	Yold.F1 = Y->F1;
	
	Yold.U2 = Y->U2;
	Yold.I2 = Y->I2;
	Yold.R2 = Y->R2;
	Yold.V2 = Y->V2;
	Yold.F2 = Y->F2;
	Yold.T2 = Y->T2;
	
	Yold.U3 = Y->U3;
	Yold.I3 = Y->I3;
	Yold.R3 = Y->R3;
	Yold.V3 = Y->V3;
	Yold.F3 = Y->F3;
	Yold.T3 = Y->T3;

	model(idx, pars, pop, Yold, &k1);

	Ytemp.U1 = Yold.U1 + h*A21*k1.U1;
	Ytemp.I1 = Yold.I1 + h*A21*k1.I1;
	Ytemp.R1 = Yold.R1 + h*A21*k1.R1;
	Ytemp.V1 = Yold.V1 + h*A21*k1.V1;
	Ytemp.F1 = Yold.F1 + h*A21*k1.F1;

	Ytemp.U2 = Yold.U2 + h*A21*k1.U2;
	Ytemp.I2 = Yold.I2 + h*A21*k1.I2;
	Ytemp.R2 = Yold.R2 + h*A21*k1.R2;
	Ytemp.V2 = Yold.V2 + h*A21*k1.V2;
	Ytemp.F2 = Yold.F2 + h*A21*k1.F2;
	Ytemp.T2 = Yold.T2 + h*A21*k1.T2;

	Ytemp.U3 = Yold.U3 + h*A21*k1.U3;
	Ytemp.I3 = Yold.I3 + h*A21*k1.I3;
	Ytemp.R3 = Yold.R3 + h*A21*k1.R3;
	Ytemp.V3 = Yold.V3 + h*A21*k1.V3;
	Ytemp.F3 = Yold.F3 + h*A21*k1.F3;
	Ytemp.T3 = Yold.T3 + h*A21*k1.T3;

	model(idx, pars, pop, Ytemp, &k2);
    
	Ytemp.U1 = Yold.U1 + h*(A31*k1.U1 + A32*k2.U1);
	Ytemp.I1 = Yold.I1 + h*(A31*k1.I1 + A32*k2.I1);
	Ytemp.R1 = Yold.R1 + h*(A31*k1.R1 + A32*k2.R1);
	Ytemp.V1 = Yold.V1 + h*(A31*k1.V1 + A32*k2.V1);
	Ytemp.F1 = Yold.F1 + h*(A31*k1.F1 + A32*k2.F1);
    
	Ytemp.U2 = Yold.U2 + h*(A31*k1.U2 + A32*k2.U2);
	Ytemp.I2 = Yold.I2 + h*(A31*k1.I2 + A32*k2.I2);
	Ytemp.R2 = Yold.R2 + h*(A31*k1.R2 + A32*k2.R2);
	Ytemp.V2 = Yold.V2 + h*(A31*k1.V2 + A32*k2.V2);
	Ytemp.F2 = Yold.F2 + h*(A31*k1.F2 + A32*k2.F2);
	Ytemp.T2 = Yold.T2 + h*(A31*k1.T2 + A32*k2.T2);
    
	Ytemp.U3 = Yold.U3 + h*(A31*k1.U3 + A32*k2.U3);
	Ytemp.I3 = Yold.I3 + h*(A31*k1.I3 + A32*k2.I3);
	Ytemp.R3 = Yold.R3 + h*(A31*k1.R3 + A32*k2.R3);
	Ytemp.V3 = Yold.V3 + h*(A31*k1.V3 + A32*k2.V3);
	Ytemp.F3 = Yold.F3 + h*(A31*k1.F3 + A32*k2.F3);
	Ytemp.T3 = Yold.T3 + h*(A31*k1.T3 + A32*k2.T3);

	model(idx, pars, pop, Ytemp, &k3);
    
	Ytemp.U1 = Yold.U1 + h*(A41*k1.U1 + A42*k2.U1 + A43*k3.U1);
	Ytemp.I1 = Yold.I1 + h*(A41*k1.I1 + A42*k2.I1 + A43*k3.I1);
	Ytemp.R1 = Yold.R1 + h*(A41*k1.R1 + A42*k2.R1 + A43*k3.R1);
	Ytemp.V1 = Yold.V1 + h*(A41*k1.V1 + A42*k2.V1 + A43*k3.V1);
	Ytemp.F1 = Yold.F1 + h*(A41*k1.F1 + A42*k2.F1 + A43*k3.F1);
    
	Ytemp.U2 = Yold.U2 + h*(A41*k1.U2 + A42*k2.U2 + A43*k3.U2);
	Ytemp.I2 = Yold.I2 + h*(A41*k1.I2 + A42*k2.I2 + A43*k3.I2);
	Ytemp.R2 = Yold.R2 + h*(A41*k1.R2 + A42*k2.R2 + A43*k3.R2);
	Ytemp.V2 = Yold.V2 + h*(A41*k1.V2 + A42*k2.V2 + A43*k3.V2);
	Ytemp.F2 = Yold.F2 + h*(A41*k1.F2 + A42*k2.F2 + A43*k3.F2);
	Ytemp.T2 = Yold.T2 + h*(A41*k1.T2 + A42*k2.T2 + A43*k3.T2);
    
	Ytemp.U3 = Yold.U3 + h*(A41*k1.U3 + A42*k2.U3 + A43*k3.U3);
	Ytemp.I3 = Yold.I3 + h*(A41*k1.I3 + A42*k2.I3 + A43*k3.I3);
	Ytemp.R3 = Yold.R3 + h*(A41*k1.R3 + A42*k2.R3 + A43*k3.R3);
	Ytemp.V3 = Yold.V3 + h*(A41*k1.V3 + A42*k2.V3 + A43*k3.V3);
	Ytemp.F3 = Yold.F3 + h*(A41*k1.F3 + A42*k2.F3 + A43*k3.F3);
	Ytemp.T3 = Yold.T3 + h*(A41*k1.T3 + A42*k2.T3 + A43*k3.T3);

	model(idx, pars, pop, Ytemp, &k4);
    
	Ytemp.U1 = Yold.U1 + h*(A51*k1.U1 + A52*k2.U1 + A53*k3.U1 + A54*k4.U1);
	Ytemp.I1 = Yold.I1 + h*(A51*k1.I1 + A52*k2.I1 + A53*k3.I1 + A54*k4.I1);
	Ytemp.R1 = Yold.R1 + h*(A51*k1.R1 + A52*k2.R1 + A53*k3.R1 + A54*k4.R1);
	Ytemp.V1 = Yold.V1 + h*(A51*k1.V1 + A52*k2.V1 + A53*k3.V1 + A54*k4.V1);
	Ytemp.F1 = Yold.F1 + h*(A51*k1.F1 + A52*k2.F1 + A53*k3.F1 + A54*k4.F1);
    
	Ytemp.U2 = Yold.U2 + h*(A51*k1.U2 + A52*k2.U2 + A53*k3.U2 + A54*k4.U2);
	Ytemp.I2 = Yold.I2 + h*(A51*k1.I2 + A52*k2.I2 + A53*k3.I2 + A54*k4.I2);
	Ytemp.R2 = Yold.R2 + h*(A51*k1.R2 + A52*k2.R2 + A53*k3.R2 + A54*k4.R2);
	Ytemp.V2 = Yold.V2 + h*(A51*k1.V2 + A52*k2.V2 + A53*k3.V2 + A54*k4.V2);
	Ytemp.F2 = Yold.F2 + h*(A51*k1.F2 + A52*k2.F2 + A53*k3.F2 + A54*k4.F2);
	Ytemp.T2 = Yold.T2 + h*(A51*k1.T2 + A52*k2.T2 + A53*k3.T2 + A54*k4.T2);
    
	Ytemp.U3 = Yold.U3 + h*(A51*k1.U3 + A52*k2.U3 + A53*k3.U3 + A54*k4.U3);
	Ytemp.I3 = Yold.I3 + h*(A51*k1.I3 + A52*k2.I3 + A53*k3.I3 + A54*k4.I3);
	Ytemp.R3 = Yold.R3 + h*(A51*k1.R3 + A52*k2.R3 + A53*k3.R3 + A54*k4.R3);
	Ytemp.V3 = Yold.V3 + h*(A51*k1.V3 + A52*k2.V3 + A53*k3.V3 + A54*k4.V3);
	Ytemp.F3 = Yold.F3 + h*(A51*k1.F3 + A52*k2.F3 + A53*k3.F3 + A54*k4.F3);
	Ytemp.T3 = Yold.T3 + h*(A51*k1.T3 + A52*k2.T3 + A53*k3.T3 + A54*k4.T3);

	model(idx, pars, pop, Ytemp, &k5);
    
	Ytemp.U1 = Yold.U1 + h*(A61*k1.U1 + A62*k2.U1 + A63*k3.U1 + A64*k4.U1 + A65*k5.U1);
	Ytemp.I1 = Yold.I1 + h*(A61*k1.I1 + A62*k2.I1 + A63*k3.I1 + A64*k4.I1 + A65*k5.I1);
	Ytemp.R1 = Yold.R1 + h*(A61*k1.R1 + A62*k2.R1 + A63*k3.R1 + A64*k4.R1 + A65*k5.R1);
	Ytemp.V1 = Yold.V1 + h*(A61*k1.V1 + A62*k2.V1 + A63*k3.V1 + A64*k4.V1 + A65*k5.V1);
	Ytemp.F1 = Yold.F1 + h*(A61*k1.F1 + A62*k2.F1 + A63*k3.F1 + A64*k4.F1 + A65*k5.F1);
    
	Ytemp.U2 = Yold.U2 + h*(A61*k1.U2 + A62*k2.U2 + A63*k3.U2 + A64*k4.U2 + A65*k5.U2);
	Ytemp.I2 = Yold.I2 + h*(A61*k1.I2 + A62*k2.I2 + A63*k3.I2 + A64*k4.I2 + A65*k5.I2);
	Ytemp.R2 = Yold.R2 + h*(A61*k1.R2 + A62*k2.R2 + A63*k3.R2 + A64*k4.R2 + A65*k5.R2);
	Ytemp.V2 = Yold.V2 + h*(A61*k1.V2 + A62*k2.V2 + A63*k3.V2 + A64*k4.V2 + A65*k5.V2);
	Ytemp.F2 = Yold.F2 + h*(A61*k1.F2 + A62*k2.F2 + A63*k3.F2 + A64*k4.F2 + A65*k5.F2);
	Ytemp.T2 = Yold.T2 + h*(A61*k1.T2 + A62*k2.T2 + A63*k3.T2 + A64*k4.T2 + A65*k5.T2);
    
	Ytemp.U3 = Yold.U3 + h*(A61*k1.U3 + A62*k2.U3 + A63*k3.U3 + A64*k4.U3 + A65*k5.U3);
	Ytemp.I3 = Yold.I3 + h*(A61*k1.I3 + A62*k2.I3 + A63*k3.I3 + A64*k4.I3 + A65*k5.I3);
	Ytemp.R3 = Yold.R3 + h*(A61*k1.R3 + A62*k2.R3 + A63*k3.R3 + A64*k4.R3 + A65*k5.R3);
	Ytemp.V3 = Yold.V3 + h*(A61*k1.V3 + A62*k2.V3 + A63*k3.V3 + A64*k4.V3 + A65*k5.V3);
	Ytemp.F3 = Yold.F3 + h*(A61*k1.F3 + A62*k2.F3 + A63*k3.F3 + A64*k4.F3 + A65*k5.F3);
	Ytemp.T3 = Yold.T3 + h*(A61*k1.T3 + A62*k2.T3 + A63*k3.T3 + A64*k4.T3 + A65*k5.T3);

	model(idx, pars, pop, Ytemp, &k6);
    
	Y->U1 = Yold.U1 + h*(A71*k1.U1 + A73*k3.U1 + A74*k4.U1 + A75*k5.U1 + A76*k6.U1);
	Y->I1 = Yold.I1 + h*(A71*k1.I1 + A73*k3.I1 + A74*k4.I1 + A75*k5.I1 + A76*k6.I1);
	Y->R1 = Yold.R1 + h*(A71*k1.R1 + A73*k3.R1 + A74*k4.R1 + A75*k5.R1 + A76*k6.R1);
	Y->V1 = Yold.V1 + h*(A71*k1.V1 + A73*k3.V1 + A74*k4.V1 + A75*k5.V1 + A76*k6.V1);
	Y->F1 = Yold.F1 + h*(A71*k1.F1 + A73*k3.F1 + A74*k4.F1 + A75*k5.F1 + A76*k6.F1);
    
	Y->U2 = Yold.U2 + h*(A71*k1.U2 + A73*k3.U2 + A74*k4.U2 + A75*k5.U2 + A76*k6.U2);
	Y->I2 = Yold.I2 + h*(A71*k1.I2 + A73*k3.I2 + A74*k4.I2 + A75*k5.I2 + A76*k6.I2);
	Y->R2 = Yold.R2 + h*(A71*k1.R2 + A73*k3.R2 + A74*k4.R2 + A75*k5.R2 + A76*k6.R2);
	Y->V2 = Yold.V2 + h*(A71*k1.V2 + A73*k3.V2 + A74*k4.V2 + A75*k5.V2 + A76*k6.V2);
	Y->F2 = Yold.F2 + h*(A71*k1.F2 + A73*k3.F2 + A74*k4.F2 + A75*k5.F2 + A76*k6.F2);
	Y->T2 = Yold.T2 + h*(A71*k1.T2 + A73*k3.T2 + A74*k4.T2 + A75*k5.T2 + A76*k6.T2);
    
	Y->U3 = Yold.U3 + h*(A71*k1.U3 + A73*k3.U3 + A74*k4.U3 + A75*k5.U3 + A76*k6.U3);
	Y->I3 = Yold.I3 + h*(A71*k1.I3 + A73*k3.I3 + A74*k4.I3 + A75*k5.I3 + A76*k6.I3);
	Y->R3 = Yold.R3 + h*(A71*k1.R3 + A73*k3.R3 + A74*k4.R3 + A75*k5.R3 + A76*k6.R3);
	Y->V3 = Yold.V3 + h*(A71*k1.V3 + A73*k3.V3 + A74*k4.V3 + A75*k5.V3 + A76*k6.V3);
	Y->F3 = Yold.F3 + h*(A71*k1.F3 + A73*k3.F3 + A74*k4.F3 + A75*k5.F3 + A76*k6.F3);
	Y->T3 = Yold.T3 + h*(A71*k1.T3 + A73*k3.T3 + A74*k4.T3 + A75*k5.T3 + A76*k6.T3);

	return;
}
// Hereeeeeeeeeeeeeeeeeeeee
// Create a new Structure for the data?
//-------------------------------------------------------------------------
__global__ void costFunction(param pars, float *pop, viralData *Vdata,
							 window *Twindows2, window *Twindows3, float *costFn)
{
	int ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= pars.Np) return;

	int penaltyFlag = 0;
	int rssFlag = 1;
	int qFlag2 = pars.qFlag;
	int qFlag3 = pars.qFlag;
	int nn = 0, qnn2 = 0, qnn3 = 0;
	int nData = pars.nData;
	int qnData2 = pars.qnData2;
	int qnData3 = pars.qnData3;

	float aux, sum2 = 0.0f;
	viralData qtData = Vdata[0];
	window qlWindow2 = Twindows2[0];
	window qlWindow3 = Twindows3[0];

	comp Y;
	int idx = ind*pars.D;
	float t = pars.t0;
	float h = pars.dt;

	// Initial values
	Y.U1 = pars.U10;
	Y.I1 = pars.I10;
	Y.R1 = pars.R10;
	Y.V1 = pow(10, pop[idx]); // V10
	Y.F1 = pars.F10;

	Y.U2 = pars.U20;
	Y.I2 = pars.I20;
	Y.R2 = pars.R20;
	Y.V2 = pars.V20;
	Y.F2 = pars.F20;
	Y.T2 = pars.T20;

	Y.U3 = pars.U30;
	Y.I3 = pars.I30;
	Y.R3 = pars.R30;
	Y.V3 = pars.V30;
	Y.F3 = pars.F30;
	Y.T3 = pars.T30;

	while (t <= pars.tN)
	{
		// Dormand-Prince method to compute the next state
		deriv_step(idx, pars, pop, &Y);
		t += h;

		// Check for NaN and INF values
		if (isnan(Y.U1) || isnan(Y.U2) || isnan(Y.U3) ||
			isnan(Y.I1) || isnan(Y.I2) || isnan(Y.I3) ||
			isnan(Y.R1) || isnan(Y.R2) || isnan(Y.R3) ||
			isnan(Y.V1) || isnan(Y.V2) || isnan(Y.V3) ||
			isnan(Y.F1) || isnan(Y.F2) || isnan(Y.F3) ||
					isnan(Y.T2) || isnan(Y.T3) ||
			isinf(Y.U1) || isinf(Y.U2) || isinf(Y.U3) ||
			isinf(Y.I1) || isinf(Y.I2) || isinf(Y.I3) ||
			isinf(Y.R1) || isinf(Y.R2) || isinf(Y.R3) ||
			isinf(Y.V1) || isinf(Y.V2) || isinf(Y.V3) ||
			isinf(Y.F1) || isinf(Y.F2) || isinf(Y.F3) ||
						   isinf(Y.T2) || isinf(Y.T3) )
		{
			penaltyFlag = 1;
			break;
		}

		if (Y.U1 < 0.0) Y.U1 = 0.0;
		if (Y.I1 < 0.0) Y.I1 = 0.0;
		if (Y.R1 < 0.0) Y.R1 = 0.0;
		if (Y.V1 < 0.0) Y.V1 = 0.0;
		if (Y.F1 < 0.0) Y.F1 = 0.0;

		if (Y.U2 < 0.0) Y.U2 = 0.0;
		if (Y.I2 < 0.0) Y.I2 = 0.0;
		if (Y.R2 < 0.0) Y.R2 = 0.0;
		if (Y.V2 < 0.0) Y.V2 = 0.0;
		if (Y.F2 < 0.0) Y.F2 = 0.0;
		if (Y.T2 < 0.0) Y.T2 = 0.0;

		if (Y.U3 < 0.0) Y.U3 = 0.0;
		if (Y.I3 < 0.0) Y.I3 = 0.0;
		if (Y.R3 < 0.0) Y.R3 = 0.0;
		if (Y.V3 < 0.0) Y.V3 = 0.0;
		if (Y.F3 < 0.0) Y.F3 = 0.0;
		if (Y.T3 < 0.0) Y.T3 = 0.0;

		// This part calculates the quantitative RSS
		if (t >= qtData.time && rssFlag)
		{
			while (1)
			{
				aux = Y.V1 < 1.0 ? 0.0 : log10(Y.V1);
				aux -= qtData.V1;
				sum2 += aux*aux;

				aux = Y.V2 < 1.0 ? 0.0 : log10(Y.V2);
				aux -= qtData.V1;
				sum2 += aux*aux;

				aux = Y.V3 < 1.0 ? 0.0 : log10(Y.V3);
				aux -= qtData.V3;
				sum2 += aux*aux;

				nn++;

				if (nn >= nData)
				{
					rssFlag = 0;
					break;
				}

				if (Vdata[nn].time != qtData.time)
				{
					qtData = Vdata[nn];
					break;
				}
			}
		}

		// This part define the penalties according to the window constraints
		// TRACHEA
		if (t > qlWindow2.minTime && qFlag2)
		{
			if (Y.T2 > qlWindow2.minValue && Y.T2 < qlWindow2.maxValue)
			{
				qnn2++;
				if (qnn2 >= qnData2) qFlag2 = 0;
				else qlWindow2 = Twindows2[qnn2];
			}
			else if (t > qlWindow2.maxTime)
			{
				penaltyFlag = 1;
				break;
			}
		}

		// LUNGS
		if (t > qlWindow3.minTime && qFlag3)
		{
			if (Y.T3 > qlWindow3.minValue && Y.T3 < qlWindow3.maxValue)
			{
				qnn3++;
				if (qnn3 >= qnData3) qFlag3 = 0;
				else qlWindow3 = Twindows3[qnn3];
			}
			else if (t > qlWindow3.maxTime)
			{
				penaltyFlag = 1;
				break;
			}
		}

		if (!rssFlag && !qFlag2 && !qFlag3) break;
	}

	if (isinf(sum2)) penaltyFlag = 1;
	costFn[ind] = penaltyFlag ? 1e38 : sum2;

	return;
}

//-------------------------------------------------------------------------------

__global__ void newPopulation(int Np, int D, float Cr, float Fm, float *randUni,
int3 *iiMut, float *lowerLim, float *upperLim, float *pop, float *newPop)
{
	int ind, jj, idx, auxInt, flag = 0;
	int3 iiM, idxM;
	float trial, auxL, auxU;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= Np) return;

	iiM = iiMut[ind];

	for (jj=0; jj<D; jj++)
	{
		idx = ind*D + jj;

		auxL = lowerLim[jj];
		auxU = upperLim[jj];
		if (auxL == auxU)
		{
			newPop[idx] = auxL;
			continue;
		}

		idxM.x = iiM.x*D + jj;
		idxM.y = iiM.y*D + jj;
		idxM.z = iiM.z*D + jj;

		if (randUni[idx] <= Cr)
		{
			// DE/rand/1 || DE/best/1
			trial = pop[idxM.x] + Fm*(pop[idxM.y] - pop[idxM.z]);
    		// DE/current-to-best/1
			//trial = pop[idx] + Fm*(pop[idxM.x] - pop[idx]) + Fm*(pop[idxM.y] - pop[idxM.z]);

			if (trial < auxL) trial = auxL;
			if (trial > auxU) trial = auxU;

			newPop[idx] = trial;
			flag = 1;
		}
		else newPop[idx] = pop[idx];
	}

	// Ensure there be at least one element
	// of the mutant vector in the new population
	if (!flag)
	{
		auxInt = ind*D;
		while (1)
		{
			jj = int(D*randUni[auxInt%(Np*D)]);
			if (jj == D) jj--;
			auxL = lowerLim[jj];
			auxU = upperLim[jj];
			if (auxL == auxU) continue;
			break;
		}

		idx = ind*D + jj;
		idxM.x = iiM.x*D + jj;
		idxM.y = iiM.y*D + jj;
		idxM.z = iiM.z*D + jj;

		// DE/rand/1 || DE/best/1
		trial = pop[idxM.x] + Fm*(pop[idxM.y] - pop[idxM.z]);
    	// DE/current-to-best/1
		//trial = pop[idx] + Fm*(pop[idxM.x] - pop[idx]) + Fm*(pop[idxM.y] - pop[idxM.z]);

		if (trial < auxL) trial = auxL;
		if (trial > auxU) trial = auxU;

		newPop[idx] = trial;
	}

	return;
}

//-------------------------------------------------------------------------------

__global__ void selection(int Np, int D, float *pop, float *newPop,
float *costFn, float *newCostFn)
{
	int ind, jj, idx;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= Np) return;

	if  (newCostFn[ind] > costFn[ind]) return;

	for (jj=0; jj<D; jj++)
	{
		idx = ind*D + jj;
		pop[idx] = newPop[idx];
	}
	costFn[ind] = newCostFn[ind];

	return;
}

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- MAIN =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

int main()
{
	/*+*+*+*+*+ FETCH DATA	+*+*+*+*+*/
	char renglon[200], dirData[500], *linea;
	FILE *fileRead;
	int nData, nn;
	float auxfloat;
	float *timeData, *meanN, *stdN, *meanT, *stdT, *meanL, *stdL;

	sprintf(dirData, "viral_load.csv");
	fileRead = fopen(dirData, "r");

	nData = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		nData++;
	}
	rewind(fileRead);

	if (nData == 0)
	{
		printf("Error: Empty file in %s\n", dirData);
		exit (1);
	}
	nData--; //Because the header line

	timeData = (float *) malloc(nData*sizeof(float));
	meanN = (float *) malloc(nData*sizeof(float));
	stdN = (float *) malloc(nData*sizeof(float));
	meanT = (float *) malloc(nData*sizeof(float));
	stdT = (float *) malloc(nData*sizeof(float));
	meanL = (float *) malloc(nData*sizeof(float));
	stdL = (float *) malloc(nData*sizeof(float));

	// Discard first line (header)
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, ",");
		sscanf(linea, "%f", &auxfloat);
		timeData[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		meanN[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		stdN[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		meanT[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		stdT[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		meanL[nn] = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		stdL[nn] = auxfloat;

		nn++;
	}
	fclose(fileRead);

	// Read qualitative constraints
	// TRACHEA
	int qnData2;
	window *Twindows2;

	sprintf(dirData, "cd8_h3n2_T.csv");
	fileRead = fopen(dirData, "r");

	qnData2 = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		qnData2++;
	}
	rewind(fileRead);

	if (qnData2 == 0)
	{
		printf("Error: Empty file in %s\n", dirData);
		exit (1);
	}
	qnData2--; // Header

	cudaMallocManaged(&Twindows2, qnData2*sizeof(window));

	// Discard first line (header)
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, ",");
		sscanf(linea, "%f", &auxfloat);
		Twindows2[nn].minTime = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		Twindows2[nn].maxTime = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		Twindows2[nn].minValue = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		Twindows2[nn].maxValue = auxfloat;

		nn++;
	}
	fclose(fileRead);

	// LUNGS
	int qnData3;
	window *Twindows3;

	sprintf(dirData, "cd8_h3n2_L.csv");
	fileRead = fopen(dirData, "r");

	qnData3 = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		qnData3++;
	}
	rewind(fileRead);

	if (qnData3 == 0)
	{
		printf("Error: Empty file in %s\n", dirData);
		exit (1);
	}
	qnData3--; // Header

	cudaMallocManaged(&Twindows3, qnData3*sizeof(window));

	// Discard first line (header)
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);

	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, ",");
		sscanf(linea, "%f", &auxfloat);
		Twindows3[nn].minTime = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		Twindows3[nn].maxTime = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		Twindows3[nn].minValue = auxfloat;

		linea = strtok(NULL, ",");
		sscanf(linea, "%f", &auxfloat);
		Twindows3[nn].maxValue = auxfloat;

		nn++;
	}
	fclose(fileRead);

	/*+*+*+*+*+ FETCH PARAMETERS +*+*+*+*+*/
	int Np, itMax, seed, D, bootFlag, qFlag;
	float Fm, Cr, t0, tN, dt;
	int err_flag = 0;

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	/* DE parameters */
	// Population of parameter vector
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &Np);

	// Maximum iterations
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &itMax);

	// Recombination probability
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &Cr);

	// Mutation factor
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &Fm);

	// Seed for random numbers
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &seed);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	/* Initial conditions for ODE solve */
	// Initial time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &t0);

	// Final time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &tN);

	// Step time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%f", &dt);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	/* Parameters to estimate */
	// Number of parameters to estimate
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &D);

	// Activate sampling for Bootstraping?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &bootFlag);

	// Include qualitative fit?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &qFlag);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	if (err_flag)
	{
		printf("Error: Something is wrong in the parameter file (.param)\n");
		exit (1);
	}

	param pars;

	pars.D = D;
	pars.m = 2;
	pars.t0 = t0;
	pars.tN = tN;
	pars.Np = Np;
	pars.dt = dt;
	pars.nData = nData;
	pars.qnData2 = qnData2;
	pars.qnData3 = qnData3;
	pars.qFlag = qFlag;

	// Initial values
	pars.U10 = 5e8;
	pars.I10 = 0.0;
	pars.R10 = 0.0;
	pars.F10 = 0.0;

	pars.U20 = 5e8;
	pars.I20 = 0.0;
	pars.R20 = 0.0;
	pars.V20 = 0.0;
	pars.F20 = 0.0;
	pars.T20 = 1e6;

	pars.U30 = 5e8;
	pars.I30 = 0.0;
	pars.R30 = 0.0;
	pars.V30 = 0.0;
	pars.F30 = 0.0;
	pars.T30 = 1e6;


	float *lowerLim, *upperLim, *pop;
	int ii, jj, idx;

	cudaMallocManaged(&lowerLim, D*sizeof(float));
	cudaMallocManaged(&upperLim, D*sizeof(float));

	float aux;
	float auxL, auxU;

	for (jj=0; jj<D; jj++)
	{
		if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
		else sscanf(renglon, "[%f : %f]", &auxL, &auxU);
		lowerLim[jj] = auxL;
		upperLim[jj] = auxU;

		if (auxL > auxU)
		{
			printf("Error: Invalid range in parameter %d (param file)\n", jj);
			exit (1);
		}
	}

	cudaMallocManaged(&pop, Np*D*sizeof(float));

	// Inicializa números aleatorios
	if (seed < 0) seed *= -1;
	Ran ranUni(seed);
	
	// Initialize population
	for (jj=0; jj<D; jj++)
	{
		aux = upperLim[jj] - lowerLim[jj];
		for (ii=0; ii<Np; ii++)
		{
			idx = ii*D + jj;
			if (aux == 0.0) pop[idx] = lowerLim[jj];
			else pop[idx] = lowerLim[jj] + aux*ranUni.doub();
		}
	}

	// Bootstraping
	viralData *Vdata;

	if (bootFlag)
	{
		Normaldev ranNorm(0.0, 1.0, seed); // Standard dev (Z)
		int sizeSample = 1;
		pars.sizeSample = sizeSample;

		// Generate random data in normal distribution
		cudaMallocManaged(&Vdata, sizeSample*nData*sizeof(viralData));

		// Linear transformation from Z to normal dev X
		// Z = (X - meanX) / stdX -> X = Z*stdX + meanX
		for (ii=0; ii<nData; ii++)
			for (jj=0; jj<sizeSample; jj++)
			{
				idx = jj + ii*sizeSample;
				Vdata[idx].time = timeData[ii];
				Vdata[idx].V1 = meanN[ii] + stdN[ii]*ranNorm.dev();
				Vdata[idx].V2 = meanT[ii] + stdT[ii]*ranNorm.dev();
				Vdata[idx].V3 = meanL[ii] + stdL[ii]*ranNorm.dev();
			}
	}
	else
	{
		cudaMallocManaged(&Vdata, nData*sizeof(viralData));

		for (ii=0; ii<nData; ii++)
		{
			Vdata[ii].time = timeData[ii];
			Vdata[ii].V1 = meanN[ii];
			Vdata[ii].V2 = meanT[ii];
			Vdata[ii].V3 = meanL[ii];
		}
	}

	free(timeData);
	free(meanN);
	free(meanT);
	free(meanL);
	free(stdN);
	free(stdT);
	free(stdL);

	int ths, blks;
	float *costFn, *d_newCostFn;

	cudaMallocManaged(&costFn, Np*sizeof(float));
	cudaMalloc(&d_newCostFn, Np*sizeof(float));

	// Estimate the number of threads and blocks for the GPU
	ths = (Np < THS_MAX) ? nextPow2(Np) : THS_MAX;
	blks = 1 + (Np - 1)/ths;

	// Cost function value for the first generation
	costFunction<<<blks, ths>>>(pars, pop, Vdata, Twindows2, Twindows3, costFn);
	cudaDeviceSynchronize();

    	/*+*+*+*+*+ START OPTIMIZATION +*+*+*+*+*/
	int it, xx, yy, zz;
	int3 *iiMut;
	float *d_randUni, *d_newPop;
	float minVal;
	int iiMin;
	curandGenerator_t gen;

	cudaMallocManaged(&iiMut, Np*sizeof(int3));
	cudaMalloc(&d_newPop, Np*D*sizeof(float));

	// Initialize random numbers with a standard normal distribution
	// I use cuRand libraries 
	cudaMalloc(&d_randUni, Np*D*sizeof(float)); // Array only for GPU
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen, seed);

	// Empiezan las iteraciones
	for (it=0; it<itMax; it++)
	{
		//minVal = valCostFn[0];
		//iiMin = 0;
		//if (!flag)
		//	for(ii=1; ii<Np; ii++) if (minVal > valCostFn[ii])
		//	{
		//		minVal = valCostFn[ii];
		//		iiMin = ii;
		//	}

		//xx = iiMin;
		for (ii=0; ii<Np; ii++)
		{
			do xx = Np*ranUni.doub(); while (xx == ii);
			do yy = Np*ranUni.doub(); while (yy == ii || yy == xx);
			do zz = Np*ranUni.doub(); while (zz == ii || zz == yy || zz == xx);

			iiMut[ii].x = xx; iiMut[ii].y = yy; iiMut[ii].z = zz;
		}

		// Generate random numbers and then update positions
		curandGenerateUniform(gen, d_randUni, Np*D);

		// Generate new population
		newPopulation<<<blks, ths>>>(Np, D, Cr, Fm, d_randUni, iiMut, lowerLim, upperLim, pop, d_newPop);

		// Calculate cost function values
		costFunction<<<blks, ths>>>(pars, d_newPop, Vdata, Twindows2, Twindows3, d_newCostFn);

		// Select the best vectors between new ones and old ones
		selection<<<blks, ths>>>(Np, D, pop, d_newPop, costFn, d_newCostFn);

		cudaDeviceSynchronize();
	}

	// Find the minimum of the population
	minVal = costFn[0];
	iiMin = 0;
	for (ii=1; ii<Np; ii++) if (minVal > costFn[ii])
	{
		minVal = costFn[ii];
		iiMin = ii;
	}

	FILE *fBestPars;
	fBestPars = fopen("bestPars.dat", "a");
	for (jj=0; jj<D; jj++) fprintf(fBestPars, "%.4e ", pop[iiMin*D + jj]);
	fprintf(fBestPars, "%.4e\n", minVal);
	fclose(fBestPars);

	printf("FINISHED\n");

	cudaFree(Vdata);
	cudaFree(Twindows2);
	cudaFree(Twindows3);
	cudaFree(lowerLim);
	cudaFree(upperLim);
	cudaFree(iiMut);
	cudaFree(pop);
	cudaFree(d_newPop);
	cudaFree(costFn);
	cudaFree(d_newCostFn);
	cudaFree(d_randUni);
	curandDestroyGenerator(gen);

	return 0;
}
