#!/usr/bin/env python

# Copyright 2021, the University of Michigan
# Full license can be found in LICENSE

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
import cmocean
import re
import sys

# fix 96 MLT bin
BinMLT=0.25

#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------

def get_args(argv):

    help = 0
    au = 100.0
    al = -200.0
    outfile = 'fta_model_au_al'
    indir = 'inputs'
    minal = al
    maxal = al
    dal = 50.0
    
    for arg in argv:

        IsFound = 0

        if (not IsFound):

            m = re.match(r'-outfile=(.*)',arg)
            if m:
                outfile = m.group(1)
                IsFound = 1

            m = re.match(r'-indir=(.*)',arg)
            if m:
                indir = m.group(1)
                IsFound = 1

            m = re.match(r'-au=(.*)',arg)
            if m:
                au = float(m.group(1))
                IsFound = 1

            m = re.match(r'-dal=(.*)',arg)
            if m:
                dal = float(m.group(1))
                IsFound = 1

            m = re.match(r'-al=(.*)',arg)
            if m:
                al = float(m.group(1))
                if (al > 0.0):
                    al = -al
                IsFound = 1

            m = re.match(r'-minal=(.*)',arg)
            if m:
                minal = float(m.group(1))
                if (minal > 0.0):
                    minal = -al
                IsFound = 1

            m = re.match(r'-maxal=(.*)',arg)
            if m:
                maxal = float(m.group(1))
                if (maxal > 0.0):
                    maxal = -al
                IsFound = 1

            m = re.match(r'-h',arg)
            if m:
                help = 1
                IsFound = 1

    if (minal > maxal):
        temp = minal
        minal = maxal
        maxal = temp
    args = {'au': au,
            'minal':minal,
            'maxal':maxal,
            'dal':dal,
            'al':al,
            'help':help,
            'indir':indir,
            'outfile':outfile}

    return args

#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------

def interp_model(mlts0,mlats0,efs0):

    BinMLat = 0.5
    nMLTs = len(np.arange(BinMLT/2, 24, BinMLT))
    nMLats = len(np.arange(BinMLat/2+50.0, 90, BinMLat))
    nLevs = 21
    mlats = np.zeros((nMLTs,nMLats))
    mlts = np.zeros((nMLTs,nMLats))
    efs = np.zeros((nMLTs,nMLats))

    mlat_inp = np.arange(50.0+BinMLat/2, 90, BinMLat)

    for k22, k2 in enumerate(np.arange(BinMLT/2, 24, BinMLT)):

        mlts[k22,:] = k2

        for ilat0,ilat in enumerate(np.arange(BinMLat/2.0+50, 90, BinMLat)):

             mlats[:,ilat0]=ilat

    for k22, k2 in enumerate(np.arange(BinMLT/2, 24, BinMLT)):

        efs_tmp0 = efs0[k22,:]
        mlat_tmp0 = mlats0[k22,:]

        lc = mlat_tmp0>0

        if len(mlat_tmp0[lc])==0:
            continue
        mlat_tmp,efs_tmp = mlat_tmp0[lc],efs_tmp0[lc]

        mlat_inp1= mlat_inp[int((mlat_tmp[0]-50)//BinMLat):
                int((mlat_tmp[-1]-50)//BinMLat+1)]

        efs_inp = np.interp(mlat_inp1, mlat_tmp, efs_tmp)

        efs[k22,int((mlat_tmp[0]-50)//BinMLat):
                int((mlat_tmp[-1]-50)//BinMLat+1)] = efs_inp

    return mlts,mlats,efs

def cal_avee(efs_lbhl,efs_lbhs):

    nMLTs = 96
    nMLats = 80
    ratio = np.ones((nMLTs,nMLats))*np.nan
    avee = np.ones((nMLTs,nMLats))*np.nan

    loc = (efs_lbhs>=1.0)&(efs_lbhl>=1.0)
    tmp = efs_lbhl[loc]/efs_lbhs[loc]
    ratio[loc] = tmp

    # Germay et al.(1994) ratio -> energy flux
    a = 0.09193196
    b = 19.73989114
    c = 0.5446197
    avee[loc] = 10**(
            np.log((ratio[loc]-c)/a)/np.log(b))
    return avee

def plot_sph(mlts,mlats,efs0,ax,mini,maxi,nls,cmap):

    efs=np.zeros((96,80))
    efs[efs0==efs0] = efs0[efs0==efs0]

    theta = mlts*15.0*np.pi/180.0-np.pi/2
    rad = 90.0- mlats

    dtheta = 0.25*15*np.pi/180.0
    wrp_theta = np.concatenate((theta,theta[-1:] + dtheta))
    wrp_E = np.concatenate((efs, efs[0:1, :]), axis=0)
    wrp_r = np.concatenate((rad,rad[0:1, :]), axis=0)

    #loc = (efs==efs)
    #efs_tmp,theta_tmp,rad_tmp,mlts_tmp=efs[loc],theta[loc],rad[loc],mlts[loc]
    #loc1 = efs_tmp.argmax()
    #theta_l = theta_tmp[loc1]
    #rad_l = rad_tmp[loc1]
    hs= ax.contourf(wrp_theta,wrp_r,wrp_E,nls,
            vmin = mini,vmax = maxi,
            cmap = cmap,
            alpha=0.85)
    #hs1 = ax.scatter(theta_l,rad_l,
    #        facecolors='none',
    #        edgecolors='m',
    #        marker = '^',alpha = 0.8)

    levels = [0,10,20,30,40]
    ax.set_rmax((40.0))
    ax.set_rlabel_position(22.5)
    ax.set_xticks(np.arange(0,np.pi*2,np.pi/4))
    ax.set_xticklabels(['', '', '12', '', '18', '', '',''])
    ax.set_rticks(levels)
    ax.set_yticklabels(['','','','','50\xb0'])
    ax.grid(True,linestyle='--')

    return

def plot_car(mlts,mlats,efs0,ax,mini,maxi,nls,cmap):

    efs=np.zeros((96,80))
    efs[efs0==efs0] = efs0[efs0==efs0]

    theta_d = (mlts+12)%24
    rad_d = mlats

    hs= ax.tricontourf(theta_d[efs==efs],
            rad_d[efs==efs],
            efs[efs==efs],nls,
            vmin = mini,vmax = maxi,
            cmap = cmap,
            alpha=0.85)

    ax.set_xlim(0,24)
    ax.set_ylim(50,90)
    ax.set_xticks(np.arange(0,24,6))
    ax.set_yticks(np.arange(50,90,5))
    #ax.set_yticklabels(['','','','',''])
    ax.set_yticklabels(['50','','60','','70','','80',''])
    #ax.yaxis.set_tick_params(pad=0.1)
    #ax.tick_params(axis='y',length=0)
    ax.set_xticklabels(['12','18','24','06'])
    ax.grid(True,linestyle='--',alpha=0.7)
    ax.set_xlabel('MLT (Hour)')
    ax.set_ylabel('MLat (Deg)')

    return

def get_factors_iaual_csv(AUs,ALs_n,
        emis_type,al0):

    ALs = -ALs_n

    forder,param = 'r1','k_k'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.csv')
    k_k = pd.read_csv(ifile)

    forder,param = 'r1','k_b'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.csv')
    k_b = pd.read_csv(ifile)

    forder,param = 'r1','b_k'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.csv')
    b_k = pd.read_csv(ifile)

    forder,param = 'r1','b_b'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.csv')
    b_b = pd.read_csv(ifile)


    forder,param = 'r2','k_k'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.csv')
    k_k2 = pd.read_csv(ifile)

    forder,param = 'r2','k_b'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.csv')
    k_b2 = pd.read_csv(ifile)

    MLTs = np.arange(BinMLT/2, 24, BinMLT)
    nMLTs = len(MLTs)
    nLevs = 21


    mlts0 = np.ones((nMLTs,nLevs))*np.nan
    mlats0 = np.ones((nMLTs,nLevs))*np.nan
    efs0 = np.ones((nMLTs,nLevs))*np.nan

    for k11,k1 in enumerate(MLTs):

        mlts0[k11,:]=k1

    kk_lat = np.asarray(k_k.iloc[:,np.arange(1,42,2)])
    kb_lat = np.asarray(k_b.iloc[:,np.arange(1,42,2)])
    bk_lat = np.asarray(b_k.iloc[:,np.arange(1,42,2)])
    bb_lat = np.asarray(b_b.iloc[:,np.arange(1,42,2)])

    kk_ef = np.asarray(k_k.iloc[:,np.arange(2,43,2)])
    kb_ef = np.asarray(k_b.iloc[:,np.arange(2,43,2)])
    bk_ef = np.asarray(b_k.iloc[:,np.arange(2,43,2)])
    bb_ef = np.asarray(b_b.iloc[:,np.arange(2,43,2)])

    #
    kk_lat2 = np.asarray(k_k2.iloc[:,np.arange(1,42,2)])
    kb_lat2 = np.asarray(k_b2.iloc[:,np.arange(1,42,2)])

    kk_ef2  = np.asarray(k_k2.iloc[:,np.arange(2,43,2)])
    kb_ef2  = np.asarray(k_b2.iloc[:,np.arange(2,43,2)])

    if (ALs <= al0):

        print('interpolate...')
        cf_b_lat = bb_lat+bk_lat*AUs
        cf_k_lat = kb_lat+kk_lat*np.log(AUs)

        cf_b_ef = bb_ef+bk_ef*AUs
        cf_k_ef = kb_ef+kk_ef*np.log(AUs)

        mlat_p = cf_b_lat + cf_k_lat * ALs
        ef_p   = cf_b_ef + cf_k_ef * ALs

    else:
        print('extrapolate...')
        # extrapolation
        cf_b_lat = bb_lat+bk_lat*AUs
        cf_k_lat = kb_lat+kk_lat*np.log(AUs)

        cf_b_ef = bb_ef+bk_ef*AUs
        cf_k_ef = kb_ef+kk_ef*np.log(AUs)

        mlat_b0 = cf_b_lat + cf_k_lat * al0
        ef_b0   = cf_b_ef + cf_k_ef * al0

        #
        cf_k_lat2 = kb_lat2+kk_lat2*AUs
        cf_k_ef2  = kb_ef2 +kk_ef2 *AUs

        mlat_p = mlat_b0 + cf_k_lat2 * (ALs-al0)
        ef_p   = ef_b0   + cf_k_ef2 * (ALs-al0)

    mlats0 = mlat_p
    efs0 = ef_p

    return mlts0,mlats0,efs0

def load_fta_aual_csv(au,al,outfile):

    plt.style.use('default')
    cmap = mpl.cm.get_cmap("inferno")

    fig1 = plt.figure(1)
    gs1 = fig1.add_gridspec(2,2)
    plt.subplots_adjust(wspace = 0.08,hspace = 0.15)

    bandtype='lbhl'
    mlts0,mlats0,efs0 = get_factors_iaual_csv(au,al,bandtype,500)
    mlts,mlats,lbhl_inp = interp_model(mlts0,mlats0,efs0)

    bandtype='lbhs'
    mlts0,mlats0,efs0 = get_factors_iaual_csv(au,al,bandtype,500)
    mlts,mlats,lbhs_inp = interp_model(mlts0,mlats0,efs0)

    eflux = lbhl_inp/110.0
    avee = cal_avee(lbhl_inp,lbhs_inp)

    efs = eflux
    mini = 0
    maxi = np.round(np.max(eflux))
    nls = int(maxi)
    cmap = mpl.cm.get_cmap("inferno")

    dlat = mlats[0,1]-mlats[0,0]
    dmlt = mlts[1,0]-mlts[0,0]
    m_per_deg = (6372.0 + 110.0) * 1000.0 * 2 * 3.14159 / 360.0
    area = dlat * m_per_deg * dmlt * 15.0 * m_per_deg * np.cos(mlats*3.1415/180.0)
    power = eflux/1000.0 * area
    hp = np.sum(power)/1.0e9

    ax=fig1.add_subplot(gs1[0,0])
    plot_car(mlts,mlats,efs,ax,mini,maxi,nls,cmap)

    ax.text(0.1,0.9, (
        'AU: {:3d} nT'.format(int(au))+'; '+
        'AL: {:3d} nT'.format(int(al))+'; '+
        'HP: {:3d} GW'.format(int(hp))),
        transform=ax.transAxes,
        color='cyan',
        fontsize=8)
    ax=fig1.add_subplot(gs1[0,1],polar=True)
    plot_sph(mlts,mlats,efs,ax,mini,maxi,nls,cmap)

    cbarax1 = fig1.add_axes([0.87,0.52,0.01,0.22])
    cbar = mpl.colorbar.ColorbarBase(cbarax1,
            cmap=cmap,
            label='Eflux (erg/cm\u00b2/s)',
            norm=mpl.colors.Normalize(mini,maxi))
    cbar.set_ticks(np.linspace(mini,maxi,3))

    efs = avee
    mini = 0
    maxi = 8
    nls = 21
    ax=fig1.add_subplot(gs1[1,0])
    plot_car(mlts,mlats,efs,ax,mini,maxi,nls,cmap)
    ax=fig1.add_subplot(gs1[1,1],polar=True)
    plot_sph(mlts,mlats,efs,ax,mini,maxi,nls,cmap)

    cbarax1 = fig1.add_axes([0.87,0.12,0.01,0.22])
    cbar = mpl.colorbar.ColorbarBase(cbarax1,
            cmap=cmap,
            label='Avee (keV)',
            norm=mpl.colors.Normalize(mini,maxi))
    cbar.set_ticks(np.linspace(mini,maxi,3))

    fig1.savefig(outfile+'_{:03d}_{:03d}.png'.format(
        int(au),int(al)),dpi=600)

    plt.close()
    
    return hp

if __name__ == '__main__':

    global DataDir

    args = get_args(sys.argv)

    if (args["help"]):

        print('Usage : ')
        print('fta_model_aual2.py -au=au -al=al -outfile=outfile.png')
        print('   -help : print this message')
        print('   -au : upper auroral index (between 0 - 300 nT)')
        print('   -al : lower auroral index (below 0, ~0 -> -1000 nT)')
        print('   -outfile : output file name (au, al, .png is added!)')
    
    DataDir=args['indir']+'/' # put data directory here
    outdir='./' # output image directory here

    if (args['minal'] == args['maxal']):
        hp = load_fta_aual_csv(args['au'],args['al'],args['outfile'])
        print('Hemispheric Power : ',hp,' GW')
    else:
        AllHp = []
        AllAl = []
        for al in np.arange(args['minal'],args['maxal'],args['dal']):
            AllHp.append(load_fta_aual_csv(args['au'],al,args['outfile']))
            AllAl.append(al)

        fig = plt.figure(1,figsize=(6,9))

        ax = plt.subplot(1,1,1)
        ax.plot(AllAl, AllHp)
        plt.title('Hemispheric Power for AU: {:3d} nT'.format(int(args['au'])))
        plt.xlabel('AL (nT)')
        plt.ylabel('Hemispheric Power (GW)')
        plt.savefig('test.png')
