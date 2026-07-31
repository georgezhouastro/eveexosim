#!/usr/bin/env python
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
class Occ(object):
    def __init__(self, stellartype):
        self.occ0 = np.loadtxt("kunimoto_table_%s.txt" % stellartype) 
        self.err1 = np.loadtxt("kunimoto_table_%s_lower.txt" % stellartype) 
        self.err2 = np.loadtxt("kunimoto_table_%s_upper.txt" % stellartype) 
        self.Parr = np.array([0.78,1.56,3.31,6.25,12.5,25,50,100,200,400,800,1600])
        self.Rarr = np.array([0.5,0.71,1.0,1.41,2.0,2.83,4.0,5.66,8.00,11.31,16])[::-1]
        self.occ  = np.zeros(self.occ0.shape)
        #print("initialize MK OCC for stellar type %s" % stellartype)

    def sample(self):
        for i in range(self.occ0.shape[0]):
            for j in range(self.occ0.shape[1]):
                if self.occ0[i,j] ==0:
                    draw = np.abs(np.random.normal())
                    # think about how to do that properly
                    #while draw>2:
                    #    draw = np.abs(np.random.normal())
                    self.occ[i,j] = draw/2.*self.err2[i,j]
                else:
                    draw = np.random.uniform()
                    if draw>0.5:
                        draw2 = np.abs(np.random.normal())
                        self.occ[i,j] = self.occ0[i,j]+draw2*self.err2[i,j]
                    else:
                        draw2 = np.abs(np.random.normal())
                        tmpocc = self.occ0[i,j]-draw2*self.err1[i,j]
                        if tmpocc<0:
                            tmpocc = 0.0
                        self.occ[i,j] = tmpocc
        return
    
    def cal_cum(self):
        self.occ_P = np.array([0]+list(np.sum(self.occ,axis=0)))/100.
        self.cum_P = np.cumsum(self.occ_P)
        self.r_cum_arr = []
        for i in range(self.occ.shape[1]):
            r_occ = np.array([0] + list(self.occ[:, i][::-1]/np.sum(self.occ[:, i])))
            r_cum = np.cumsum(r_occ)
            self.r_cum_arr.append(r_cum)
        #print(np.max(self.cum_P), np.min(self.cum_P)) 
        return

    def draw(self):
        
        # draw period 
        pdraws = []
        rpdraws = []
        Nplanet = int(np.max(self.cum_P))+1
        planetcount = 0
        for i in range(Nplanet):
            draw = np.random.random()#*np.max(self.cum_P)
            if i<(Nplanet-1):
                planetflag = True
                indexdraw = np.where((self.cum_P/np.max(self.cum_P))<=draw)[0][-1]
            else:
                #print(draw,np.max(self.cum_P)-i) 
                if draw > (np.max(self.cum_P)-i):
                    planetflag = False 
                    break
                else:
                    planetflag = True
                indexdraw = np.where((self.cum_P-i)<=draw)[0][-1]
            if indexdraw ==0:
                pdraw = 10**((np.log10(self.Parr[indexdraw])-np.log10(0.1)) *np.random.random()+np.log10(0.1)) 
            else:
                pdraw = 10**((np.log10(self.Parr[indexdraw])-np.log10(self.Parr[indexdraw-1])) *np.random.random()+np.log10(self.Parr[indexdraw-1])) 
            # draw radius 
            if planetflag:
                indexdraw = np.where(self.cum_P<=draw)[0][-1]
                draw = np.random.random()
                r_cum = self.r_cum_arr[indexdraw]
                indexdraw = np.where(r_cum<=draw)[0][-1]
                rdraw = 10**((np.log10(self.Rarr[indexdraw+1])-np.log10(self.Rarr[indexdraw])) *np.random.random()+np.log10(self.Rarr[indexdraw])) 
                Rp = rdraw/110.
            else:
                Rp = 0
                pdraw = 20
            #print draw, np.max(cumm_hj), pdraw, Rp
            #inc = np.random.random()
            
            rpdraw = Rp
            pdraws.append(pdraw)
            rpdraws.append(rpdraw)
            #RJ = 0.102
            #if Rp>0.6*RJ:
            #    Rp = (np.random.normal()*0.2+1.)*RJ
            #    while (Rp<0.6*RJ or Rp>2.2*RJ):
            #        Rp = (np.random.normal()*0.2+1.)*RJ

            if planetflag:
                planetcount+=1
        #print(Nplanet, planetcount)

        return [pdraws, rpdraws]

if __name__=='__main__':
    F_occ = Occ("G")
    nsize = 1000
    ps = []
    rps = []
    for i in range(1000):
        F_occ.sample()
        F_occ.cal_cum()
        p,rp = F_occ.draw()
        ps+= p
        rps+= rp
    print(len(np.array(ps)), len(np.array(rps)))

    plt.scatter(np.array(ps), np.array(rps))
    plt.loglog()
    plt.show()
    sys.exit()
    import pandas as pd
    df = pd.DataFrame({"period":np.array(ps), "radius":np.array(rps)})
    df.to_csv("test_Fdraw.csv")

