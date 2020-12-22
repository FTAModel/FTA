#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
# import cmocean

# fix 96 MLT bin
BinMLT=0.25

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

def plot_1dail(mlt,mlat,var,maxi,mini,ax,nls):

    BinAE = 50
    #cmapnew = mpl.cm.get_cmap("BuPu")
    cmapnew = mpl.cm.get_cmap("plasma")
    theta = mlt*15.0*np.pi/180.0-np.pi/2
    rad = 90.0- mlat

    dtheta = 0.25*15*np.pi/180.0
    wrp_theta = np.concatenate((theta,theta[-1:] + dtheta))
    wrp_E = np.concatenate((var, var[0:1, :]), axis=0)
    wrp_r = np.concatenate((rad,rad[0:1, :]), axis=0)

    loc = (var==var)
    var,theta,rad=var[loc],theta[loc],rad[loc]

    hs= ax.contourf(wrp_theta,wrp_r,wrp_E,nls,
           vmin = mini, vmax = maxi, cmap=cmapnew)

    return

def read_emission(ddir,AE_median,emis_type):

    BinAE = 50
    k1 = AE_median
    data=pd.read_csv((ddir+(
        'AE_{:03d}_{:03d}_'+emis_type+
        '_fta_emis.csv').format(int(k1-BinAE/2),
            int(k1+BinAE/2))),header=None)
    data1=pd.read_csv((ddir+(
        'AE_{:03d}_{:03d}_'+emis_type+
        '_fta_mlt.csv').format(int(k1-BinAE/2),
            int(k1+BinAE/2))),header=None)
    data2=pd.read_csv((ddir+(
        'AE_{:03d}_{:03d}_'+emis_type+
        '_fta_mlat.csv').format(int(k1-BinAE/2),
            int(k1+BinAE/2))),header=None)

    emis = np.asarray(data)
    mlts = np.asarray(data1)
    mlats = np.asarray(data2)

    return emis, mlts, mlats

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

def fta_model_load(ddir,save_eflux,save_avee):

    ###---------------------------------------------------------
    # dail plots of energy flux and average energy for each AE bin
    # Input : 1. ddir (directory of LBHl and LBHs data)
    #         2. save_eflux (dir and name to save energy flux)
    #         3. save_avee (dir and name to save average energy)
    # contain: 1. read_emission -> read LBHl and s data for each AE bin
    #          2. interp_model -> interpolate 96 x 21 bins data with
    #             0.25 h MLT bin and 0.5 deg MLT bin from 50-90 deg
    #          3. cal_avee -> calculate average energy based on LBHl
    #             and LBHs data
    #          4. plot_1dail -> plot one dail plot for var
    # Chen Wu (12/22/2020)
    ###---------------------------------------------------------


    plt.style.use('default')

    ### you can change colormap to 'plasma'
    #cmapnew = cmocean.cm.dense
    cmapnew = mpl.cm.get_cmap("plasma")
    #cmapnew = mpl.cm.get_cmap("BuPu")

    AEmax = 900
    BinAE = 50
    BinMLat = 0.5
    BinMLT = 0.25

    nMLTs = len(np.arange(BinMLT/2, 24, BinMLT))
    nMLats = len(np.arange(BinMLat/2+50.0, 90, BinMLat))
    mlts = np.ones((nMLTs,nMLats))*np.nan
    mlats = np.ones((nMLTs,nMLats))*np.nan

    for k22, k2 in enumerate(np.arange(BinMLT/2, 24, BinMLT)):

        mlts[k22,:] = k2

        for ilat0,ilat in enumerate(np.arange(BinMLat/2.0+50, 90, BinMLat)):

             mlats[:,ilat0]=ilat
    for k11,k1 in enumerate(np.arange(BinAE/2.0, AEmax, BinAE)):

        if k11 ==0:
            fig_ef1 = plt.figure(1)
            gs_ef1 = fig_ef1.add_gridspec(3,3)
            plt.subplots_adjust(wspace = 0.02,hspace = 0.02)

            fig_eg1 = plt.figure(3)
            gs_eg1 = fig_eg1.add_gridspec(3,3)
            plt.subplots_adjust(wspace = 0.02,hspace = 0.02)

        elif k11 == 9:
            fig_ef2 = plt.figure(2)
            gs_ef2 = fig_ef2.add_gridspec(3,3)
            plt.subplots_adjust(wspace = 0.02,hspace = 0.02)

            fig_eg2 = plt.figure(4)
            gs_eg2 = fig_eg2.add_gridspec(3,3)
            plt.subplots_adjust(wspace = 0.02,hspace = 0.02)

        ### set maxi and mini of dial plots here
        #  maxi_ef and mini_ef for energy flux
        #  maxi_eg and mini_eg for average energy
        #  maxi1_ef and mini1_ef for colorbar in plot 1 of eflux
        #  ...
        if k11<9:
            maxi_ef = 10
            maxi_eg = 8
            mini = 0

            maxi1_ef = maxi_ef
            mini1_ef = mini
            maxi1_eg = maxi_eg
            mini1_eg = mini

            ax_ef=fig_ef1.add_subplot(gs_ef1[k11],polar = True)
            ax_eg=fig_eg1.add_subplot(gs_eg1[k11],polar = True)

        elif (k11 <=17 and k11>8):
            maxi_ef = 15
            maxi_eg = 8
            mini = 0

            maxi2_ef = maxi_ef
            mini2_ef = mini
            maxi2_eg = maxi_eg
            mini2_eg = mini

            ax_ef=fig_ef2.add_subplot(gs_ef2[k11-9],polar = True)
            ax_eg=fig_eg2.add_subplot(gs_eg2[k11-9],polar = True)


        # read lbhl emission for this AE bin (median=k1,interval=50 nT)
        lbhl_19,mlt_19,mlat_19 = read_emission(ddir,k1,'lbhl')

        ### eflux = lbhl/110.0
        mlts,mlats,lbhl_inp = interp_model(mlt_19,mlat_19,lbhl_19)
        eflux = lbhl_inp/110.0

        ### avee
        # 1. interpotate LBHl and LBHs
        # 2. call cal_avee
        lbhs_19,mlt_19s,mlat_19s = read_emission(ddir,k1,'lbhs')
        mlts,mlats,lbhs_inp = interp_model(mlt_19s,mlat_19s,lbhs_19)

        avee = cal_avee(lbhl_inp,lbhs_inp)

        #### plotting
        ### plot energy flux pattern for this AE bin
        nls1 = 8
        #plot_1dail(mlts,mlats,eflux,maxi_ef,mini,
        #        ax_ef,nls1)  #plot with interpotated data

        plot_1dail(mlt_19,mlat_19,lbhl_19/100.0,maxi_ef,mini,
                ax_ef,nls1)   #plot with 19-level data

        ### plot average energy pattern for this AE bin
        nls1 = 11
        plot_1dail(mlts,mlats,avee,maxi_eg,mini,
                ax_eg,nls1)

        theta = mlts*15.0*np.pi/180.0-np.pi/2
        rad = 90.0- mlats

        loc = eflux==eflux
        efs_tmp,theta_tmp,rad_tmp=eflux[loc],theta[loc],rad[loc]

        loc1 = efs_tmp.argmax()
        theta_l = theta_tmp[loc1]
        rad_l = rad_tmp[loc1]

        hs1 = ax_ef.scatter(theta_l,rad_l,
                facecolors='none',
                edgecolors='orange',
                marker = '^',alpha = 0.9)
        ax_ef.text(-0.08,0.70,
                (
                    '{:03d}'.format(int(k1-BinAE/2.0))+
                    '-'+
                    '{:03d} nT'.format(int(k1+BinAE/2.0))),
                transform=ax_ef.transAxes,
                rotation = 45,
                   fontsize=7)
        ax_eg.text(-0.08,0.70,
                (
                    '{:03d}'.format(int(k1-BinAE/2.0))+
                    '-'+
                    '{:03d} nT'.format(int(k1+BinAE/2.0))),
                transform=ax_eg.transAxes,
                rotation = 45,
                   fontsize=7)
        levels = [0,10,20,30,35]
        ax_ef.set_rmax((35.0))
        ax_ef.set_rlabel_position(22.5)
        ax_ef.set_xticks(np.linspace(0,2*np.pi,5))
        ax_ef.set_xticklabels(['', '', '', '',''])
        ax_ef.set_yticks(np.array([0,10,20,30,35]))
        ax_ef.set_yticklabels(['','','','','55\xb0'],fontsize=7)
        ax_ef.set_rticks(levels)
        ax_ef.grid(True,linestyle='--')

        levels = [0,10,20,30,35]
        ax_eg.set_rmax((35.0))
        ax_eg.set_rlabel_position(22.5)
        ax_eg.set_xticks(np.linspace(0,2*np.pi,5))
        ax_eg.set_xticklabels(['', '', '', '',''])
        ax_eg.set_yticks(np.array([0,10,20,30,35]))
        ax_eg.set_yticklabels(['','','','','55\xb0'],fontsize=7)
        ax_eg.set_rticks(levels)
        ax_eg.grid(True,linestyle='--')

    # add colorbar to eflux plot
    cbarax1 = fig_ef1.add_axes([0.90,0.12,0.02,0.2])
    cbar = mpl.colorbar.ColorbarBase(cbarax1,
            cmap=cmapnew,
            label='Eflux (erg/cm\u00b2/s)',
            norm=mpl.colors.Normalize(mini1_ef,maxi1_ef))
    cbar.set_ticks(np.linspace(mini1_ef,maxi1_ef,3))

    cbarax1 = fig_ef2.add_axes([0.90,0.12,0.02,0.2])
    cbar = mpl.colorbar.ColorbarBase(cbarax1,
            cmap=cmapnew,
            label='Eflux (erg/cm\u00b2/s)',
            norm=mpl.colors.Normalize(mini2_ef,maxi2_ef))
    cbar.set_ticks(np.linspace(mini2_ef,maxi2_ef,3))


    # add colorbar to avee plot
    cbarax1 = fig_eg1.add_axes([0.90,0.12,0.02,0.2])
    cbar = mpl.colorbar.ColorbarBase(cbarax1,
            cmap=cmapnew,
            label='Avee (keV)',
            norm=mpl.colors.Normalize(mini1_ef,maxi1_ef))
    cbar.set_ticks(np.linspace(mini1_ef,maxi1_ef,3))

    cbarax1 = fig_eg2.add_axes([0.90,0.12,0.02,0.2])
    cbar = mpl.colorbar.ColorbarBase(cbarax1,
            cmap=cmapnew,
            label='Avee (keV)',
            norm=mpl.colors.Normalize(mini2_eg,maxi2_eg))
    cbar.set_ticks(np.linspace(mini2_eg,maxi2_eg,3))

    fig_ef1.savefig(save_eflux[0:-4]+'_a.png',dpi=600)
    fig_ef2.savefig(save_eflux[0:-4]+'_b.png',dpi=600)
    fig_eg1.savefig(save_avee[0:-4]+'_a.png',dpi=600)
    fig_eg2.savefig(save_avee[0:-4]+'_b.png',dpi=600)

    # plt.show()

    return

if __name__ == '__main__':


    DataDir='./inputs/' # put data directory here

    # dir and filename of plot to save
    save_eflux = './eflux_test1222.png' # for energy flux
    save_avee = './avee_test1222.png' # for avee

    fta_model_load(DataDir,save_eflux,save_avee)



