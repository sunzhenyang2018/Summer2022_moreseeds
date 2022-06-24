#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _GFluct_reg();
extern void _IKa_reg();
extern void _IMmintau_reg();
extern void _Ih_reg();
extern void _Ikdrf_reg();
extern void _Ipasssd_reg();
extern void _Nasoma_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," GFluct.mod");
fprintf(stderr," IKa.mod");
fprintf(stderr," IMmintau.mod");
fprintf(stderr," Ih.mod");
fprintf(stderr," Ikdrf.mod");
fprintf(stderr," Ipasssd.mod");
fprintf(stderr," Nasoma.mod");
fprintf(stderr, "\n");
    }
_GFluct_reg();
_IKa_reg();
_IMmintau_reg();
_Ih_reg();
_Ikdrf_reg();
_Ipasssd_reg();
_Nasoma_reg();
}
