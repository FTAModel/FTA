#!/usr/bin/env python

# Copyright 2021, the University of Michigan
# Full license can be found in LICENSE

# The AU/AL model here is kept in step with GITM's Fortran version,
# GITMCode/Electrodynamics src/ModFtaModel.f90
# Last synced at 756d8cb (2026-07-25).
# To compare the two, see validate/README.md.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import re
import sys

# fix 96 MLT bin
BinMLT=0.25

# fix 0.5 deg MLat bins from MinMLat to 90 (same grid as GITM's ModFtaModel.f90)
BinMLat=0.5
MinMLat=30.0

# lowest MLat shown in the plots
PlotMinMLat=50.0

# |AL| above which the second (log_4p) fit is used
ALSplit=500.0

# average energy outside of the oval (GITM uses 2.0 keV; nan masks it in plots)
AveEFill=np.nan

# how the 21 energy bins are put onto the MLat grid:
#   'gitm'  : bin-average, then linearly fill empty bins (as in GITM)
#   'numpy' : np.interp
LatInterp='gitm'

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def get_args(argv):

    help = 0
    fre = 0
    au = 100.0
    al = -200.0
    outfile = 'fta_model_au_al'
    indir = 'inputs_aual'
    minal = al
    maxal = al
    dal = 50.0
    noaafile = 'hpke.noaa'
    hp = 50.0
    interp = LatInterp

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

            m = re.match(r'-noaafile=(.*)',arg)
            if m:
                noaafile = m.group(1)
                IsFound = 1

            m = re.match(r'-hp=(.*)',arg)
            if m:
                hp = float(m.group(1))
                IsFound = 1

            m = re.match(r'-interp=(.*)',arg)
            if m:
                interp = m.group(1)
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
                    minal = -minal
                IsFound = 1

            m = re.match(r'-maxal=(.*)',arg)
            if m:
                maxal = float(m.group(1))
                if (maxal > 0.0):
                    maxal = -maxal
                IsFound = 1

            m = re.match(r'(-h|-help|--help)$',arg)
            if m:
                help = 1
                IsFound = 1

            m = re.match(r'-fre',arg)
            if m:
                fre = 1
                IsFound = 1

    if (interp not in ['gitm','numpy']):
        print('-interp must be gitm or numpy, not : ',interp)
        exit()

    if (minal > maxal):
        temp = minal
        minal = maxal
        maxal = temp

    args = {'au': au,
            'minal':minal,
            'maxal':maxal,
            'dal':dal,
            'al':al,
            'hp':hp,
            'interp':interp,
            'help':help,
            'fre':fre,
            'indir':indir,
            'noaafile':noaafile,
            'outfile':outfile}

    return args

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def interp_model(mlts0,mlats0,efs0,interp=LatInterp):

    nMLTs = len(np.arange(BinMLT/2, 24, BinMLT))
    nMLats = len(np.arange(BinMLat/2+MinMLat, 90, BinMLat))
    nLevs = 21
    mlats = np.zeros((nMLTs,nMLats))
    mlts = np.zeros((nMLTs,nMLats))
    efs = np.zeros((nMLTs,nMLats))

    mlat_inp = np.arange(MinMLat+BinMLat/2, 90, BinMLat)

    for k22, k2 in enumerate(np.arange(BinMLT/2, 24, BinMLT)):

        mlts[k22,:] = k2

        for ilat0,ilat in enumerate(np.arange(BinMLat/2.0+MinMLat, 90, BinMLat)):

             mlats[:,ilat0]=ilat

    nUnsorted = 0

    for k22, k2 in enumerate(np.arange(BinMLT/2, 24, BinMLT)):

        efs_tmp0 = efs0[k22,:]
        mlat_tmp0 = mlats0[k22,:]

        if (interp == 'gitm'):

            # average the energy bins that fall in each MLat bin:
            for ilat0,ilat in enumerate(mlat_inp):
                lc = ((mlat_tmp0 > ilat-BinMLat/2) &
                      (mlat_tmp0 <= ilat+BinMLat/2))
                if np.sum(lc) > 0:
                    efs[k22,ilat0] = np.mean(efs_tmp0[lc])

            # then linearly fill the empty MLat bins inside the oval:
            idx = np.where(efs[k22,:] > 0)[0]
            if len(idx)==0:
                continue
            for i in range(idx[0]+1, idx[-1]):
                if efs[k22,i] == 0:
                    ii = idx[idx > i][0]
                    efs[k22,i] = (efs[k22,i-1]-efs[k22,ii])*(i-ii)/(i-1-ii) + \
                        efs[k22,ii]

        else:

            lc = mlat_tmp0>0

            if len(mlat_tmp0[lc])==0:
                continue
            mlat_tmp,efs_tmp = mlat_tmp0[lc],efs_tmp0[lc]

            # np.interp needs increasing latitudes
            if np.any(np.diff(mlat_tmp) < 0):
                nUnsorted = nUnsorted + 1
                srt = np.argsort(mlat_tmp)
                mlat_tmp,efs_tmp = mlat_tmp[srt],efs_tmp[srt]

            i0 = max(int((mlat_tmp[0]-MinMLat)//BinMLat), 0)
            i1 = min(int((mlat_tmp[-1]-MinMLat)//BinMLat+1), nMLats)

            efs[k22,i0:i1] = np.interp(mlat_inp[i0:i1], mlat_tmp, efs_tmp)

    if (nUnsorted > 0):
        print('  np.interp: sorted latitudes in ',nUnsorted,' MLT sectors')

    return mlts,mlats,efs

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def cal_avee(efs_lbhl,efs_lbhs):

    ratio = np.ones(efs_lbhl.shape)*np.nan
    avee = np.ones(efs_lbhl.shape)*AveEFill

    # Germay et al.(1994) ratio -> energy flux
    a = 0.09193196
    b = 19.73989114
    c = 0.5446197

    loc = (efs_lbhs>1.0)&(efs_lbhl>1.0)
    tmp = efs_lbhl[loc]/efs_lbhs[loc]
    ratio[loc] = tmp
    loc = loc & (ratio > c)

    avee[loc] = 10**(
            np.log((ratio[loc]-c)/a)/np.log(b))
    return avee

#-----------------------------------------------------------------------------
# General Plotting for Polar Grids
#-----------------------------------------------------------------------------

def general_polar_plot(mlts2d, mlats2d, values, ax, mini, maxi, cmap):

    rad = 90.0 - mlats2d
    theta = mlts2d * 15 * np.pi / 180.0 - np.pi/2.0

    # Do we need to wrap around in MLT?
    if (mlts2d[-1,0] % 24 != 0.0):
        theta = np.concatenate((theta,theta[0:1,:]), axis = 0)
        values = np.concatenate((values, values[0:1,:]), axis= 0)
        rad = np.concatenate((rad,rad[0:1,:]), axis= 0)

    nLevels = 31
    hs = ax.contourf(theta, rad, values, nLevels,
                     vmin = mini,vmax = maxi,
                     cmap = cmap,
                     alpha = 0.85)

    rLevels = [0,10,20,30,40]
    ax.set_rmax((np.max(rLevels)))
    ax.set_rticks(rLevels)
    ax.set_rlabel_position(22.5)
    ax.set_xticks(np.arange(0,np.pi*2,np.pi/4))
    ax.set_xticklabels(['', '', '12', '', '18', '', '',''])
    ax.set_yticklabels(['','','','','50\xb0'])
    ax.grid(True,linestyle='--')

    return

#-----------------------------------------------------------------------------
# General Plotting for Cartesian Grids
#-----------------------------------------------------------------------------

def general_cart_plot(mlts2d, mlats2d, values, ax, mini, maxi, cmap):

    nLevels = 31

    hs = ax.contourf(mlts2d, mlats2d,
            values, nLevels,
            vmin = mini,vmax = maxi,
            cmap = cmap, alpha=0.85)

    ax.set_xlim(0,24)
    ax.set_ylim(50,90)
    ax.set_xticks(np.arange(0,30,6))
    ax.set_yticks(np.arange(50,90,5))
    ax.set_yticklabels(['50','','60','','70','','80',''])
    #ax.set_xticklabels(['12','18','24','06'])
    ax.grid(True,linestyle='--',alpha=0.7)
    ax.set_xlabel('MLT (Hour)')
    ax.set_ylabel('MLat (Deg)')

    return

#-----------------------------------------------------------------------------
# plot eflux and avee in both cartesian and polar coordinates
#-----------------------------------------------------------------------------

def plot2x2(mlts2d, mlats2d, eFlux, AveE, outfile):

    #au = inputs['au']
    #al = inputs['au']
    #hp = inputs['hp']

    plt.style.use('default')
    cmap = mpl.colormaps["inferno"]
    fig1 = plt.figure(1)
    gs1 = fig1.add_gridspec(2,2)
    plt.subplots_adjust(wspace = 0.08,hspace = 0.15)
    cmap = mpl.colormaps["inferno"]

    # EFlux First:

    mini = 0
    maxi = np.round(np.max(eFlux*10.0))/10.0

    ax=fig1.add_subplot(gs1[0,0])
    general_cart_plot(mlts2d, mlats2d, eFlux, ax, mini, maxi, cmap)

    #ax.text(0.1,0.9, (
    #    'AU: {:3d} nT'.format(int(au))+'; '+
    #    'AL: {:3d} nT'.format(int(al))+'; '+
    #    'HP: {:3d} GW'.format(int(hp))),
    #    transform=ax.transAxes,
    #    color='cyan',
    #    fontsize=8)
    hp = calculate_hemispheric_power(mlats2d, mlts2d, eFlux)
    ax.text(0.1,0.9, (
        'HP: {:3d} GW'.format(int(hp))),
        transform=ax.transAxes,
        color='cyan',
        fontsize=8)

    ax=fig1.add_subplot(gs1[0,1],polar=True)
    general_polar_plot(mlts2d, mlats2d, eFlux, ax, mini, maxi, cmap)

    cbarax1 = fig1.add_axes([0.87,0.52,0.01,0.22])
    cbar = mpl.colorbar.ColorbarBase(cbarax1,
            cmap=cmap,
            label='Eflux (erg/cm\u00b2/s)',
            norm=mpl.colors.Normalize(mini,maxi))
    cbar.set_ticks(np.linspace(mini,maxi,3))

    # AveE Second:

    mini = 0
    maxi = np.round(np.max(AveE*10.0))/10.0

    ax=fig1.add_subplot(gs1[1,0])
    general_cart_plot(mlts2d, mlats2d, AveE, ax, mini, maxi, cmap)

    ax=fig1.add_subplot(gs1[1,1], polar=True)
    general_polar_plot(mlts2d, mlats2d, AveE, ax, mini, maxi, cmap)

    cbarax1 = fig1.add_axes([0.87,0.12,0.01,0.22])
    cbar = mpl.colorbar.ColorbarBase(cbarax1,
            cmap=cmap,
            label='Avee (keV)',
            norm=mpl.colors.Normalize(mini,maxi))
    cbar.set_ticks(np.linspace(mini,maxi,3))

    print('Writing file : ',outfile)
    fig1.savefig(outfile, dpi=600)

    exit()

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def plot_sph(mlts,mlats,efs0,ax,mini,maxi,nls,cmap):

    efs=np.zeros(efs0.shape)
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

    levels = list(np.arange(0,90-PlotMinMLat+1,10))
    ax.set_rmax((90-PlotMinMLat))
    ax.set_rlabel_position(22.5)
    ax.set_xticks(np.arange(0,np.pi*2,np.pi/4))
    ax.set_xticklabels(['', '', '12', '', '18', '', '',''])
    ax.set_rticks(levels)
    ax.set_yticklabels(['']*(len(levels)-1)+['{:2d}\xb0'.format(int(90-levels[-1]))])
    ax.grid(True,linestyle='--')

    return

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def plot_car(mlts,mlats,efs0,ax,mini,maxi,nls,cmap):

    efs=np.zeros(efs0.shape)
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
    ax.set_ylim(PlotMinMLat,90)
    ax.set_xticks(np.arange(0,24,6))
    ax.set_yticks(np.arange(PlotMinMLat,90,5))
    #ax.yaxis.set_tick_params(pad=0.1)
    #ax.tick_params(axis='y',length=0)
    ax.set_xticklabels(['12','18','24','06'])
    ax.grid(True,linestyle='--',alpha=0.7)
    ax.set_xlabel('MLT (Hour)')
    ax.set_ylabel('MLat (Deg)')

    return

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def get_factors_iaual(AUs,ALs_n,
        emis_type,al0):

    ALs = -ALs_n

    forder,param = 'r1','k_k'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.txt')
    k_k = np.loadtxt(ifile)

    forder,param = 'r1','k_b'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.txt')
    k_b = np.loadtxt(ifile)

    forder,param = 'r1','b_k'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.txt')
    b_k = np.loadtxt(ifile)

    forder,param = 'r1','b_b'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'.txt')
    b_b = np.loadtxt(ifile)


    forder,param = 'r2','k_k'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'_log_4p.txt')
    k_k2 = np.loadtxt(ifile)

    forder,param = 'r2','k_b'
    ifile = (DataDir+'fit_coef_21bins_'+emis_type+'_'+forder+'_'+param+'_log_4p.txt')
    k_b2 = np.loadtxt(ifile)

    MLTs = np.arange(BinMLT/2, 24, BinMLT)
    nMLTs = len(MLTs)
    nLevs = 21


    mlts0 = np.ones((nMLTs,nLevs))*np.nan
    mlats0 = np.ones((nMLTs,nLevs))*np.nan
    efs0 = np.ones((nMLTs,nLevs))*np.nan

    for k11,k1 in enumerate(MLTs):

        mlts0[k11,:]=k1

    kk_lat = np.asarray(k_k[:,np.arange(1,42,2)])
    kb_lat = np.asarray(k_b[:,np.arange(1,42,2)])
    bk_lat = np.asarray(b_k[:,np.arange(1,42,2)])
    bb_lat = np.asarray(b_b[:,np.arange(1,42,2)])

    kk_ef = np.asarray(k_k[:,np.arange(2,43,2)])
    kb_ef = np.asarray(k_b[:,np.arange(2,43,2)])
    bk_ef = np.asarray(b_k[:,np.arange(2,43,2)])
    bb_ef = np.asarray(b_b[:,np.arange(2,43,2)])

    kk_lat2 = np.asarray(k_k2[:,np.arange(1,42,2)])
    kb_lat2 = np.asarray(k_b2[:,np.arange(1,42,2)])

    kk_ef2  = np.asarray(k_k2[:,np.arange(2,43,2)])
    kb_ef2  = np.asarray(k_b2[:,np.arange(2,43,2)])

    if (ALs < al0):

        cf_b_lat = bb_lat+bk_lat*AUs
        cf_k_lat = kb_lat+kk_lat*np.log(AUs)

        cf_b_ef = bb_ef+bk_ef*AUs
        cf_k_ef = kb_ef+kk_ef*np.log(AUs)

        mlat_p = cf_b_lat + cf_k_lat * ALs
        ef_p   = cf_b_ef + cf_k_ef * ALs

    else:
        # extrapolation
        cf_b_lat = bb_lat+bk_lat*AUs
        cf_k_lat = kb_lat+kk_lat*np.log(AUs)

        cf_b_ef = bb_ef+bk_ef*AUs
        cf_k_ef = kb_ef+kk_ef*np.log(AUs)

        mlat_b0 = cf_b_lat + cf_k_lat * al0
        ef_b0   = cf_b_ef + cf_k_ef * al0

        #
        cf_k_lat2 = kb_lat2+kk_lat2*np.log(AUs)
        cf_k_ef2  = kb_ef2 +kk_ef2 *np.log(AUs)
        cf_k_ef2[cf_k_ef2 < 0] = 0.0

        mlat_p = mlat_b0 + cf_k_lat2 * (ALs-al0)
        ef_p   = ef_b0   + cf_k_ef2 * (ALs-al0)

    mlats0 = mlat_p
    efs0 = ef_p

    return mlts0,mlats0,efs0

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def calculate_hemispheric_power(mlats, mlts, eflux):

    # Calculate the hemispheric power from the pattern:
    dlat = mlats[0,1]-mlats[0,0]
    dmlt = mlts[1,0]-mlts[0,0]
    if (dlat == 0):
        dlat = mlats[1,0]-mlats[0,0]
        dmlt = mlts[0,1]-mlts[0,0]
    m_per_deg = (6372.0 + 110.0) * 1000.0 * 2 * 3.14159 / 360.0
    area = dlat * m_per_deg * dmlt * 15.0 * m_per_deg * \
        np.cos(mlats*3.1415/180.0)
    power = eflux/1000.0 * area
    hp = np.sum(power)/1.0e9

    return hp

def limit_al_lower(au):

    # most negative AL allowed for a given AU (from GITM's ModFtaModel.f90)

    if au < 150.0:
        a = -7.89e-06
        b = -0.0952
        c = -66.0
        al = (-np.sqrt(b**2 - 4.0*(c - au)*a) - b)/(2.0*a) + 10.0

    elif au <= 175.0:
        al1 = -3100.0
        al2 = -3025.0
        al = (al2 - al1)/(175.0 - 150.0)*(au - 150.0) + al1 + 10.0

    elif au <= 300.0:
        coef1 = 0.18503781
        coef2 = 2.37717756
        au_ = (au - 175.0)/125.0
        al = np.arcsinh(au_/coef1)/coef2 * 1275.0 - 3010.0

    elif au <= 575.0:
        coef1 = 0.27380323
        coef2 = 2.0135317
        au_ = (au - 325.0)/250.0
        al = np.arcsinh(au_/coef1)/coef2 * 300.0 - 1580.0

    elif au <= 775.0:
        al = (au - 1290.0)/0.5573

    else:
        al = (au - 1430.0)/0.689

    return al

def limit_al_upper(au):

    # least negative AL allowed for a given AU > 750 (from GITM's ModFtaModel.f90)

    return (au - 724.0)/(-0.682)

def limit_aual(au,al):

    au1 = au
    if au1 > 1050:
        au1 = 1050.0
    if au1 < 25:
        au1 = 25.0

    al1 = al
    al_lower = limit_al_lower(au1)
    if al1 < al_lower:
        al1 = al_lower

    if au1 > 750:
        al_upper = limit_al_upper(au1)
        if al1 > al_upper:
            al1 = al_upper

    return au1,al1

#-----------------------------------------------------------------------------
# Make sure that each band is at least 0.25 deg wide, with the energy bins
# at least 0.25/20 deg apart (otherwise the band can have negative width)
#-----------------------------------------------------------------------------

def adjust_width(mlats0):

    nLevs = mlats0.shape[1]
    dMin = 0.25/20.0
    loc = (mlats0[:,-1] - mlats0[:,0]) < 0.25

    for k11 in np.where(loc)[0]:
        for k22 in range(nLevs-1):
            if (mlats0[k11,k22+1] - mlats0[k11,k22]) < dMin:
                mlats0[k11,k22+1] = mlats0[k11,k22] + dMin

    return mlats0

#-----------------------------------------------------------------------------
# Where the lbhl and lbhs bands overlap in MLT, move both onto common
# equatorward and poleward boundaries, blending in from the edges of the
# overlap. Each band is stretched between the new boundaries.
#-----------------------------------------------------------------------------

def adjust_offset(mlts0, mlats0_l, mlats0_s, overlap):

    mlts = mlts0[:,0]

    gap1 = np.min(mlts[overlap]) - BinMLT
    gap3 = np.max(mlts[overlap]) + BinMLT
    gap2 = (gap1 + gap3)*0.5

    wght = np.zeros(len(mlts))
    loc = (mlts > gap1) & (mlts <= gap2)
    wght[loc] = (mlts[loc] - gap1)/(gap2 - gap1)*0.5
    loc = (mlts > gap2) & (mlts < gap3)
    wght[loc] = (1 - (mlts[loc] - gap2)/(gap3 - gap2))*0.5

    eb = mlats0_s[:,0]*wght + mlats0_l[:,0]*(1.0 - wght)
    pb = mlats0_s[:,-1]*wght + mlats0_l[:,-1]*(1.0 - wght)
    width_s = mlats0_s[:,-1] - mlats0_s[:,0]
    width_l = mlats0_l[:,-1] - mlats0_l[:,0]

    loc = (mlts > gap1) & (mlts < gap3)

    for k11 in np.where(loc)[0]:
        mlats0_s[k11,1:-1] = (mlats0_s[k11,1:-1] - mlats0_s[k11,0]) * \
            (pb[k11] - eb[k11])/width_s[k11] + eb[k11]
        mlats0_l[k11,1:-1] = (mlats0_l[k11,1:-1] - mlats0_l[k11,0]) * \
            (pb[k11] - eb[k11])/width_l[k11] + eb[k11]
        mlats0_s[k11,0] = eb[k11]
        mlats0_s[k11,-1] = pb[k11]
        mlats0_l[k11,0] = eb[k11]
        mlats0_l[k11,-1] = pb[k11]

    return mlats0_l, mlats0_s

#-----------------------------------------------------------------------------
# Calculate the lbhl, lbhs, eflux, avee and polar cap patterns
#-----------------------------------------------------------------------------

def calc_fta_aual(au,al,interp=LatInterp):

    # limit au&al
    au_tmp,al_tmp = limit_aual(au,al)
    if (au_tmp != au) or (al_tmp != al):
        print('au&al limited to:', au_tmp,al_tmp)

    bandtype='lbhl'
    mlts0,mlats0_l,efs0_l = get_factors_iaual(au_tmp,al_tmp,bandtype,ALSplit)

    bandtype='lbhs'
    mlts0,mlats0_s,efs0_s = get_factors_iaual(au_tmp,al_tmp,bandtype,ALSplit)

    mlats0_l = adjust_width(mlats0_l)
    mlats0_s = adjust_width(mlats0_s)

    # lbhs equatorward boundary poleward of the lbhl poleward boundary:
    overlap = (mlats0_s[:,0] - mlats0_l[:,-1]) >= -0.25/20.0
    if overlap.any():
        mlats0_l,mlats0_s = adjust_offset(mlts0,mlats0_l,mlats0_s,overlap)

    # lbhl equatorward boundary poleward of the lbhs poleward boundary:
    overlap = (mlats0_l[:,0] - mlats0_s[:,-1]) >= -0.25/20.0
    if overlap.any():
        mlats0_l,mlats0_s = adjust_offset(mlts0,mlats0_l,mlats0_s,overlap)

    mlts,mlats,lbhl_inp = interp_model(mlts0,mlats0_l,efs0_l,interp)
    mlts,mlats,lbhs_inp = interp_model(mlts0,mlats0_s,efs0_s,interp)

    # polar cap is poleward of the lbhs poleward boundary
    polarcap = (mlats > mlats0_s[:,-1:]) * 1.0

    eflux = lbhl_inp/110.0
    avee = cal_avee(lbhl_inp,lbhs_inp)

    fta = {'mlts':mlts, 'mlats':mlats,
           'lbhl':lbhl_inp, 'lbhs':lbhs_inp,
           'eflux':eflux, 'avee':avee, 'polarcap':polarcap,
           'au':au_tmp, 'al':al_tmp, 'ae':au_tmp-al_tmp,
           'limited':(au_tmp != au) or (al_tmp != al)}

    return fta

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def load_fta_aual(au,al,outfile,interp=LatInterp):

    fta = calc_fta_aual(au,al,interp)
    mlts,mlats = fta['mlts'],fta['mlats']
    eflux,avee = fta['eflux'],fta['avee']

    efs = eflux
    mini = 0
    maxi = np.round(np.max(eflux))
    nls = int(maxi)

    plt.style.use('default')
    cmap = mpl.colormaps["inferno"]
    fig1 = plt.figure(1)
    gs1 = fig1.add_gridspec(2,2)
    plt.subplots_adjust(wspace = 0.08,hspace = 0.15)
    cmap = mpl.colormaps["inferno"]

    hp = calculate_hemispheric_power(mlats, mlts, eflux)

    # Plot the patterns:
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

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def read_fre_data(NoaaFile):

    scale = 0.001
    print('Reading NOAA file : '+NoaaFile)
    fpin = open(NoaaFile, 'r')

    # read 4 header lines:
    fpin.readline()
    fpin.readline()
    fpin.readline()
    fpin.readline()

    Vars = ['Hall', 'Pedersen', 'AveE', 'eFlux']
    nVars = 4

    nMlts = 30
    nLats = 21

    MinLat = 50.0
    dLat = 2.0
    dMlt = 24.0 / nMlts

    nIndices = 10

    mlts = np.arange(0, 24.0+dMlt, dMlt)
    lats = np.arange(MinLat, 90.0+dLat, dLat)

    mlts2d, lats2d = np.meshgrid(mlts, lats)

    fre = {'mlts' : mlts2d, 'mlats' : lats2d}

    for var in Vars:
        fre[var] = np.zeros((nIndices, nLats, nMlts+1))

    for var in Vars:
        fpin.readline()
        print('Storing variable : '+var)

        for index in np.arange(0, nIndices):

            for iLat in np.arange(0, nLats):
                line1 = fpin.readline().strip()
                line2 = fpin.readline().strip()
                line = line1 + '  ' + line2
                line = line.split()
                vals = np.array(line).astype(float)
                fre[var][index, iLat, 0:nMlts] = vals * scale
                fre[var][index, iLat, nMlts] = fre[var][index, iLat, 0]

    hp = []
    for index in np.arange(0, nIndices):
        eflux = fre['eFlux'][index]
        hp.append(calculate_hemispheric_power(lats2d, mlts2d, eflux))
    fre['hp'] = hp

    return fre

# ----------------------------------------------------------------
# Drive Fuller-Rowel and Evans Model
# ----------------------------------------------------------------

def get_fre_patterns(ModelValues, hp):

    index = -1
    for modelhp in ModelValues['hp']:
        if (hp > modelhp):
            index = index+1
    if (index < 0):
        index = 0

    ratio = hp / ModelValues['hp'][index]

    print('Adjusting FR&E pattern from ')
    print('  Model : ', ModelValues['hp'][index])
    print('  Input : ', hp)
    print('  Ratio : ', ratio)

    AveE = ModelValues['AveE'][index,:,:]
    eFlux = ModelValues['eFlux'][index,:,:] * ratio

    return AveE, eFlux

# ----------------------------------------------------------------
# Main code:
# ----------------------------------------------------------------

if __name__ == '__main__':

    global DataDir

    args = get_args(sys.argv)

    if (args["help"]):

        d = get_args([])
        print('Usage : fta_model_aual.py [options]')
        print('   -h, -help, --help : print this message')
        print('   -au=AU            : upper auroral index, nT; limited to 25 - 1050'
              ' (default: {:g})'.format(d['au']))
        print('   -al=AL            : lower auroral index, nT; limited by an'
              ' AU-dependent range (default: {:g})'.format(d['al']))
        print('   -minal=AL         : min AL to sweep through (default: the -al value)')
        print('   -maxal=AL         : max AL to sweep through (default: the -al value)')
        print('   -dal=DAL          : delta AL to sweep min-max AL with'
              ' (default: {:g})'.format(d['dal']))
        print('   -interp=METHOD    : gitm or numpy, how energy bins go onto the'
              ' MLat grid (default: {})'.format(d['interp']))
        print('   -outfile=NAME     : output file name; _AU_AL.png is added'
              ' (default: {})'.format(d['outfile']))
        print('   -indir=DIR        : directory where input files are stored'
              ' (default: {})'.format(d['indir']))
        print('   -fre              : use the Fuller-Rowell and Evans [1987] model')
        print('   -hp=HP            : hemispheric power, GW, to drive FR&E'
              ' (default: {:g})'.format(d['hp']))
        print('   -noaafile=FILE    : file with FR&E values, in -indir'
              ' (default: {})'.format(d['noaafile']))
        exit()

    DataDir=args['indir']+'/' # put data directory here
    outdir='./' # output image directory here

    if (args['fre']):

        NoaaFile = DataDir + args['noaafile']
        FreModelValues = read_fre_data(NoaaFile)
        mlts2d = FreModelValues['mlts']
        mlats2d = FreModelValues['mlats']

        AveE, eFlux = get_fre_patterns(FreModelValues, args['hp'])
        outfile = 'fre_hp_{:03d}.png'.format(int(args['hp']))
        plot2x2(mlts2d, mlats2d, eFlux, AveE, outfile)

    else:
        if (args['minal'] == args['maxal']):
            hp = load_fta_aual(args['au'],args['al'],args['outfile'],args['interp'])
            print('Hemispheric Power : ',hp,' GW')
        else:
            AllHp = []
            AllAl = []
            for al in np.arange(args['minal'],args['maxal'],args['dal']):
                AllHp.append(load_fta_aual(args['au'],al,args['outfile'],args['interp']))
                AllAl.append(al)

            fig = plt.figure(1,figsize=(6,9))

            ax = plt.subplot(1,1,1)
            ax.plot(AllAl, AllHp)
            plt.title('Hemispheric Power for AU: {:3d} nT'.format(int(args['au'])))
            plt.xlabel('AL (nT)')
            plt.ylabel('Hemispheric Power (GW)')
            plt.savefig('test.png')
