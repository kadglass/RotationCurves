import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from matplotlib.lines import Line2D
from matplotlib import colors as clr
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, DictFormatter

from astropy.table import Table
from astropy.table import vstack as avstack
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM, z_at_value

from scipy.spatial import cKDTree, ConvexHull

from sympy import solve_poly_system, im
from sympy.abc import x,y

from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder.voidfinder_functions import xyz_to_radecz

#matplotlib.rcParams.update({'font.size': 38})

#Code written originally by Dahlia Veyrat

D2R = np.pi/180.

def toSky(cs):
    '''
    Convert Cartesian coordinates to sky coordinates
    '''
    c1  = cs.T[0]
    c2  = cs.T[1]
    c3  = cs.T[2]
    
    r   = np.sqrt(c1**2. + c2**2. + c3**2.)
    dec = np.arcsin(c3/r)/D2R
    ra  = (np.arccos(c1/np.sqrt(c1**2. + c2**2.))*np.sign(c2)/D2R)%360
    
    return r,ra,dec



def toCoord(r,ra,dec):
    '''
    Convert sky coordinates to Cartesian coordinates
    '''
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    
    return c1,c2,c3




class VoidMapVF():

    def __init__(self,gdata,vfdata,vfdata2):
        self.load_data(gdata,vfdata,vfdata2)

    def load_data(self,gdata,vfdata,vfdata2):
        self.gdata=gdata
        self.vfdata=vfdata
        self.vfdata2=vfdata2
        #making the data from galaxies more accessible/easier to manipulate?
        if np.isin('rabsmag', gdata.colnames):
            self.rmag = gdata['rabsmag']
        radecr = 0
        if np.isin('r', gdata.colnames):
            self.gr = gdata['r']
            radecr+=1
        elif np.isin('redshift', gdata.colnames):
            self.gr=  z_to_comoving_dist(np.array(gdata['redshift'],dtype=np.float32),0.315,1)
            radecr+=1
        if np.isin('ra', gdata.colnames):
            self.gra  = gdata['ra']
            radecr+=1
        if np.isin('dec', gdata.colnames):
            self.gdec = gdata['dec']
            radecr+=1
        #else:
        #    create ra, dec, r columns from x, y, z, data
        if radecr<3:
            self.gx,self.gy,self.gz = gdata['x'], gdata['y'], gdata['z']
        #Converting galaxy data to cartesian
        #cKDTree finds nearest neighbors to data point
        else:
            self.gx,self.gy,self.gz = toCoord(self.gr,self.gra,self.gdec)
        self.kdt = cKDTree(np.array([self.gx,self.gy,self.gz]).T)

        #Simplifying VoidFinder maximal sphere coordinates and converting them into RA,DEC,DIS
        self.vfx = vfdata['x'] 
        self.vfy = vfdata['y']
        self.vfz = vfdata['z']
        self.vfr,self.vfra,self.vfdec = toSky(np.array([self.vfx,self.vfy,self.vfz]).T)
        self.vfrad = vfdata['radius']
        self.vfedge = vfdata['edge']

        #Not sure what's going on here
        self.vfc  = matplotlib.cm.nipy_spectral(np.linspace(0,1,len(self.vfr)))
        self.vfcc = np.random.choice(range(len(self.vfc)),len(self.vfc),replace=False)

        #Same coordinate conversion, but for holes
        self.vflag = vfdata2['flag']
        self.vfx2 = vfdata2['x']
        self.vfy2 = vfdata2['y']
        self.vfz2 = vfdata2['z']
        self.vfr1,self.vfra1,self.vfdec1 = toSky(np.array([self.vfx2,self.vfy2,self.vfz2]).T)
        self.vfrad1 = vfdata2['radius']

        #matches holes to voids, and then matches galaxies within and outside of voids.
        self.vfx4   = [self.vfx2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfy4   = [self.vfy2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfz4   = [self.vfz2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfr2   = [self.vfr1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfra2  = [self.vfra1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfdec2 = [self.vfdec1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfrad2 = [self.vfrad1[self.vflag==vfl] for vfl in np.unique(self.vflag)]

        #Unsure, creating empty arrays with the same length as the number of voids 
        #(Specifically using the x coordinate)
        #unsure of purpose, currently
        self.gflag_vf = np.zeros(len(self.gx),dtype=bool)

        #VoidFinder - Finding all the points within a given radius of a specific point
        for vfl in np.unique(self.vflag):
            self.vfx3 = self.vfx2[self.vflag==vfl]
            self.vfy3 = self.vfy2[self.vflag==vfl]
            self.vfz3 = self.vfz2[self.vflag==vfl]
            self.vfrad3 = self.vfrad1[self.vflag==vfl]
            for i in range(len(self.vfx3)):
                galinds = self.kdt.query_ball_point([self.vfx3[i],self.vfy3[i],self.vfz3[i]],self.vfrad3[i])
                self.gflag_vf[galinds] = True

        #Marking wall galaxies as true
        self.wflag_vf = (1-self.gflag_vf).astype(bool)



    def cint2(self, dec):
        '''
        Calculate radii of hole-slice intersections
        '''
        cr = []
        for i in range(len(self.vfr2)):
            cr.append([])
            for j in range(len(self.vfr2[i])):
                #minimum distance from hole center to declination slice
                #shouldn't this use half the current angle to get an actual right triangle? (giving us dtd/2)
                dtd = np.abs(self.vfr2[i][j]*np.sin((self.vfdec2[i][j]-dec)*D2R))
                if dtd>self.vfrad2[i][j]:
                    cr[i].append(0.)
                else:
                    cr[i].append(np.sqrt(self.vfrad2[i][j]**2.-dtd**2.))
        return cr
    
    def cint2xyz(self, plane_height, vfn4):
        '''
        Calculate radii of hole-slice intersections
        plane_height (float): height plane in the x, y or z direction
        vfn4 (list): one of self.vfx4, self.vfy4, or self.vfz4, corresponding to plane_height
        '''

        #list of <list of radii of hole-plane intersections> for each void
        cr = []
        #iterate through each void
        for i in range(len(self.vfr2)):
            #create new cr entry for current void
            cr.append([])
            #iterate through holes in void
            for j in range(len(self.vfr2[i])):
                #minimum distance from hole center to the plane
                dtd = np.abs(vfn4[i][j] - plane_height)
                if dtd>self.vfrad2[i][j]:
                    cr[i].append(0.)
                else:
                    cr[i].append(np.sqrt(self.vfrad2[i][j]**2.-dtd**2.))
        return cr


    def gcp2(self, cc1,cc2,crad,npt,chkdpth,ra_dec_z=True):
        '''
        Convert circles' coordinates to ordered boundary
        '''
        if ra_dec_z:
            ccx = cc1*np.cos(cc2*D2R)
            ccy = cc1*np.sin(cc2*D2R)
        else:
            ccx,ccy = cc1,cc2

        Cx = [np.linspace(0.,2*np.pi,int(npt*crad[k]/10)) for k in range(len(ccx))]
        Cy = [np.linspace(0.,2*np.pi,int(npt*crad[k]/10)) for k in range(len(ccx))]

        Cx = [np.cos(Cx[k])*crad[k]+ccx[k] for k in range(len(ccx))]
        Cy = [np.sin(Cy[k])*crad[k]+ccy[k] for k in range(len(ccx))]

        for i in range(len(ccx)):
            for j in range(len(ccx)):
                if i==j:
                    continue
                cut = (Cx[j]-ccx[i])**2.+(Cy[j]-ccy[i])**2.>crad[i]**2.
                Cx[j] = Cx[j][cut]
                Cy[j] = Cy[j][cut]

        Cp = []

        for i in range(len(ccx)):
            Cp.extend(np.array([Cx[i],Cy[i]]).T.tolist())

        Cp = np.array(Cp)

        kdt = cKDTree(Cp)

        Cpi = [0]

        while len(Cpi)<len(Cp):
            if len(Cpi)==1:
                nid = kdt.query(Cp[Cpi[-1]],2)[1][1]
            else:
                nids = kdt.query(Cp[Cpi[-1]],chkdpth+1)[1][1:]
                for k in range(chkdpth):
                    if nids[k] not in Cpi[(-1*(chkdpth+1)):-1]:
                        nid = nids[k]
                        break
                nids = kdt.query(Cp[Cpi[-1]],7)[1][1:]
            Cpi.append(nid)

        #Cpi.append(0)
        if ra_dec_z:
            C1 = np.sqrt(Cp[Cpi].T[0]**2.+Cp[Cpi].T[1]**2.)
            C2 = (np.sign(Cp[Cpi].T[1])*np.arccos(Cp[Cpi].T[0]/C1)+np.pi*(1.-np.sign(Cp[Cpi].T[1])))/D2R

            return C1,C2
        else:
            return Cp[Cpi].T[0], Cp[Cpi].T[1]


    # Plot VoidFinder Voids (Version 2)
    def pvf2(self,dec,wdth,npc,chkdpth, 
    ra0, ra1, cz0, cz1, title, graph = None, zlimits = True, rot = 0, 
    colors = ["blue","blue","blue"], gal_colors = ["black","red"], include_gals=True, include_voids=True, alpha=0.2, border_alpha = 1,
    horiz_legend_offset=.8, plot_sdss = True, sdss_lim=332.38626, sdss_color='magenta', 
             mag_limit = None, galaxy_point_size = 1):
        '''
        Plot VoidFinder voids
        dec (float): declination of slice
        wdth (float): Distance from declination plane in Mpc/h within which galaxies are plotted
        '''
        if zlimits:
            cz0 = z_to_comoving_dist(np.array([cz0],dtype=np.float32),.315,1)[0]
            cz1 = z_to_comoving_dist(np.array([cz1],dtype=np.float32),.315,1)[0]

        if graph is None:
            fig = plt.figure(1, figsize=(1600/96,800/96))

            ax3, aux_ax3 = setup_axes3(fig, 111, ra0, ra1, cz0, cz1, rot)

            #aux_ax3.set_aspect(1) #This is causing errors

            plt.title(f"{title} $\delta$ = {dec}$^\circ$", loc='left',)
        else:
            fig, ax3, aux_ax3 = graph[0], graph[1], graph[2]
        
        if include_voids:
            Cr = self.cint2(dec)

            for i in range(len(self.vfr)):

                if np.sum(Cr[i])>0:

                    Cr2, Cra2 = self.gcp2(self.vfr2[i], self.vfra2[i], Cr[i], npc, chkdpth)

                    if self.vfedge[i] == 1:
                        vcolor = colors[0]#'gold'
                    elif self.vfedge[i] == 2:
                        vcolor = colors[1]#'darkgoldenrod'
                    else:
                        vcolor = colors[2]

                    #vcolor = 'blue'

                    aux_ax3.plot(Cra2, Cr2, color=vcolor, alpha = border_alpha)
                    aux_ax3.fill(Cra2, Cr2, color=vcolor, alpha=alpha)
                

            #Cr2,Cra2 = gcp3(vfx4[i],vfy4[i],vfz4[i],vfr2[i],vfrad2[i],vfdec2[i],dec,npc,chkdpth)
            #if len(Cr2)>0:
            #    aux_ax3.plot(Cra2,Cr2,color='blue')
            #    aux_ax3.fill(Cra2,Cr2,alpha=0.5,color='blue')
        if include_gals:
            # Edge galaxies
            #gdcut = (gr[eflag_vf]*np.sin((gdec[eflag_vf] - dec)*D2R))**2 < wdth**2
            #aux_ax3.scatter(gra[eflag_vf][gdcut], gr[eflag_vf][gdcut], color='g', s=1, alpha=0.5)

            # Wall galaxies
            gdcut = (self.gr[self.wflag_vf]*np.sin((self.gdec[self.wflag_vf] - dec)*D2R))**2. < wdth**2.
            if mag_limit is not None:
                gdcut = gdcut * self.rmag[self.wflag_vf] < mag_limit
            aux_ax3.scatter(self.gra[self.wflag_vf][gdcut], self.gr[self.wflag_vf][gdcut], color=gal_colors[0], s=galaxy_point_size, edgecolor=None)

            # Void galaxies
            gdcut = (self.gr[self.gflag_vf]*np.sin((self.gdec[self.gflag_vf] - dec)*D2R))**2. < wdth**2.
            if mag_limit is not None:
                gdcut = gdcut * self.rmag[self.gflag_vf] < mag_limit
            aux_ax3.scatter(self.gra[self.gflag_vf][gdcut], self.gr[self.gflag_vf][gdcut], color=gal_colors[1], s=galaxy_point_size, edgecolor=None)

            #survey_title = ["Uchuu Blue Control","Uchuu Blue Cut","Uchuu Red Control","Uchuu Red Cut"][survey_index]
        
        if plot_sdss:
            cc=plt.Circle((0, 0), sdss_lim, color=sdss_color,fill=False,linewidth=3)
            ax3.add_artist( cc )
            
        #create legend
        void_legend_handles=[]
        if include_voids:
            legend_color=clr.to_rgba(colors[2])
            legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
            void_legend_handles.append(
                Line2D([0], [0], label='voids', marker='o', markersize=15, 
                    markeredgecolor=colors[2], markerfacecolor=legend_color, linestyle=''
                )
            )
            if colors[1]!=colors[2]:
                legend_color=clr.to_rgba(colors[1])
                legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
                void_legend_handles.append(
                    Line2D([0], [0], label='near-edge voids', marker='o', markersize=15, 
                        markeredgecolor=colors[1], markerfacecolor=legend_color, linestyle='')
                )
            if colors[0]!=colors[2]:
                legend_color=clr.to_rgba(colors[0])
                legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
                void_legend_handles.append(
                    Line2D([0], [0], label='edge voids', marker='o', markersize=15, 
                        markeredgecolor=colors[0], markerfacecolor=legend_color, linestyle='')
                )
        if include_gals:
            
            void_legend_handles.append(
                Line2D([0], [0], label="wall galaxies", marker='o', markersize=9, 
                    markeredgecolor=gal_colors[0], markerfacecolor=gal_colors[0], linestyle='')
            )
            void_legend_handles.append(
                Line2D([0], [0], label="void galaxies", marker='o', markersize=9, 
                    markeredgecolor=gal_colors[1], markerfacecolor=gal_colors[1], linestyle='')
            )
        if plot_sdss:
            void_legend_handles.append(
                Line2D([0], [0], label='SDSS limit', color=sdss_color, linewidth=5)
            )
            
        aux_ax3.legend(handles=void_legend_handles, loc="upper left", bbox_to_anchor=(horiz_legend_offset,1))
        
        self.graph = [fig, ax3, aux_ax3]

        return self.graph
    
    # Plot VoidFinder Voids from a Cubic Simulation (Version 2)
    def pvf2xyz(self,plane_height,wdth,npc,chkdpth, 
        title, h="x",v="y",n="z", h_range = (0,50), v_range = (0,50), graph = None, 
        colors = ["blue","blue","blue"],gal_colors = ["black","red"],include_gals=True,alpha=0.2, border_alpha = 1,scale=1):
            '''
            Plot VoidFinder voids
            '''
            axes = {"x":self.vfx4, "y":self.vfy4, "z":self.vfz4}
            gal_axes = {"x":self.gx, "y":self.gy, "z":self.gz}
            vfh4 = axes[h]
            vfv4 = axes[v]
            vfn4 = axes[n]
            gh = gal_axes[h]
            gv = gal_axes[v]
            gn = gal_axes[n]
                
            if graph is None:

                v_scale = (v_range[1]-v_range[0])/(h_range[1]-h_range[0])

                fig, ax = plt.subplots(num=1, figsize=(scale*800/96, scale*v_scale*800/96))

                ax.set_xlabel(h)
                ax.set_ylabel(v)

                ax.set_xlim(h_range)
                ax.set_ylim(v_range)
                
                ax.set_aspect('equal', adjustable='box')

                plt.title(f"{title} ${n}$ = {plane_height} [Mpc/h]")
            else:
                fig, ax = graph[0], graph[1]

            Cr = self.cint2xyz(plane_height, vfn4)

            for i in range(len(self.vfr)):

                if np.sum(Cr[i])>0:

                    Cr2, Cra2 = self.gcp2(vfh4[i], vfv4[i], Cr[i], npc, chkdpth, ra_dec_z=False)

                    if self.vfedge[i] == 1:
                        vcolor = colors[0]#'gold'
                    elif self.vfedge[i] == 2:
                        vcolor = colors[1]#'darkgoldenrod'
                    else:
                        vcolor = colors[2]

                    #vcolor = 'blue'

                    ax.plot(Cra2, Cr2, color=vcolor, alpha = border_alpha)
                    ax.fill(Cra2, Cr2, color=vcolor, alpha=alpha)

                #Cr2,Cra2 = gcp3(vfx4[i],vfy4[i],vfz4[i],vfr2[i],vfrad2[i],vfdec2[i],dec,npc,chkdpth)
                #if len(Cr2)>0:
                #    aux_ax3.plot(Cra2,Cr2,color='blue')
                #    aux_ax3.fill(Cra2,Cr2,alpha=0.5,color='blue')
            if include_gals:
                # Edge galaxies
                #gdcut = (gr[eflag_vf]*np.sin((gdec[eflag_vf] - dec)*D2R))**2 < wdth**2
                #aux_ax3.scatter(gra[eflag_vf][gdcut], gr[eflag_vf][gdcut], color='g', s=1, alpha=0.5)

                # Wall galaxies
                gdcut = (gn[self.wflag_vf] - plane_height)**2. < wdth**2.
                ax.scatter(gh[self.wflag_vf][gdcut], gv[self.wflag_vf][gdcut], color=gal_colors[0], s=1)

                # Void galaxies
                gdcut = (gn[self.gflag_vf] - plane_height)**2. < wdth**2.
                ax.scatter(gh[self.gflag_vf][gdcut], gv[self.gflag_vf][gdcut], color=gal_colors[1], s=1)

            
            self.graph = [fig, ax]

            return self.graph


class VoidMapV2():

    def __init__(self,tridata, gzdata, zvdata, zbdata, gdata, edge_threshold = 0.1):
        self.load_data(tridata, gzdata, zvdata, zbdata, gdata, edge_threshold)

    def load_data(self, tridata, gzdata, zvdata, zbdata, gdata, edge_threshold):
        '''
        # Vsquared triangle output
        tridata = Table.read("DR7_triangles.dat",format='ascii.commented_header')
        gzdata = Table.read("DR7_galzones.dat",format='ascii.commented_header')
        zvdata = Table.read("DR7_zonevoids.dat",format='ascii.commented_header')
        zbdata = Table.read("DR7_zobovoids.dat",format='ascii.commented_header')
        gdata = Table.read("../galaxy_catalog/DR7.txt",format="ascii.commented_header") #galaxies
        '''
        

        #gr   = gdata[1].data['Rgal']
        if np.isin('rabsmag', gdata.colnames):
            self.rmag = gdata['rabsmag']
            
        self.gr   = z_to_comoving_dist(gdata['redshift'].astype(np.float32), 0.315, 1)
        self.gra  = gdata['ra']
        self.gdec = gdata['dec']

        self.gx,self.gy,self.gz = toCoord(self.gr,self.gra,self.gdec)

        g_z = gzdata['zone']
        z_v = zvdata['void0']
        self.edge = zbdata["edge_area"]/zbdata['tot_area'] > edge_threshold

        gflag_v2 = np.zeros(len(self.gr),dtype=bool)

        for z in range(len(z_v)):
            if z_v[z] > -1:
                gflag_v2[g_z==z] = True
                
        self.wflag_v2 = (1-gflag_v2).astype(bool)
        self.gflag_v2 = gflag_v2

        self.p1_r,self.p1_ra,self.p1_dec = toSky(np.array([tridata['p1_x'],tridata['p1_y'],tridata['p1_z']]).T)
        self.p2_r,self.p2_ra,self.p2_dec = toSky(np.array([tridata['p2_x'],tridata['p2_y'],tridata['p2_z']]).T)
        self.p3_r,self.p3_ra,self.p3_dec = toSky(np.array([tridata['p3_x'],tridata['p3_y'],tridata['p3_z']]).T)

        self.p1_x = tridata['p1_x']
        self.p1_y = tridata['p1_y']
        self.p1_z = tridata['p1_z']

        self.p2_x = tridata['p2_x']
        self.p2_y = tridata['p2_y']
        self.p2_z = tridata['p2_z']

        self.p3_x = tridata['p3_x']
        self.p3_y = tridata['p3_y']
        self.p3_z = tridata['p3_z']

        trivids = np.array(tridata['void_id'])

        v2c  = matplotlib.cm.nipy_spectral(np.linspace(0,1,np.amax(trivids)+1))
        v2cc = np.random.choice(range(len(v2c)),len(v2c),replace=False)

        self.tridat= tridata
        self.trivids = trivids
        self.v2c = v2c
        self.v2cc = v2cc





    def getinx(self, xx,aa,yy,bb,zz,cc,dd):
        negb = -1.*aa*xx-bb*yy+cc*dd*dd*zz
        sqto = 0.5*np.sqrt((2.*aa*xx+2.*bb*yy-2.*cc*dd*dd*zz)**2.-4.*(aa**2.+bb**2.-cc*cc*dd*dd)*(xx**2.+yy**2.-zz*zz*dd*dd))
        twa = aa**2.+bb**2.-cc*cc*dd*dd
        tt = (negb+sqto)/twa
        if tt>0 and tt<1:
            tt = tt
        else:
            tt = (negb-sqto)/twa
        return xx+aa*tt,yy+bb*tt,zz+cc*tt
    
    def isin2(self, p,ps):
        nc = 1
        for i in range(len(ps)-1):
            p1 = ps[i]
            p2 = ps[i+1]
            if p1[0]<p[0] and p2[0]<p[0]:
                continue
            elif (p1[1]-p[1])*(p2[1]-p[1])>0:
                continue
            elif p1[0]>p[0] and p2[0]>p[0]:
                nc = nc+1
            elif ((p2[1]-p1[1])/(p2[0]-p1[0]))*((p1[1]-p[1])-((p2[1]-p1[1])/(p2[0]-p1[0]))*(p1[0]-p[0]))<1:
                nc = nc+1
        return nc%2==0

    def trint2(self, dec):
        p1_dec,  p2_dec,  p3_dec = self.p1_dec, self.p2_dec, self.p3_dec
        p1_x,p1_y,p1_z = self.p1_x,self.p1_y,self.p1_z
        p2_x,p2_y,p2_z = self.p2_x,self.p2_y,self.p2_z
        p3_x,p3_y,p3_z = self.p3_x,self.p3_y,self.p3_z
        trivids = self.trivids
        getinx = self.getinx

        decsum = np.array([(p1_dec>dec).astype(int),(p2_dec>dec).astype(int),(p3_dec>dec).astype(int)]).T
        intr  = [[] for _ in range(np.amax(trivids)+1)]
        intra = [[] for _ in range(np.amax(trivids)+1)]
        for i in range(len(trivids)):
            if np.sum(decsum[i])==0:
                continue
            if np.sum(decsum[i])==3:
                continue
            cv = trivids[i]
            if np.sum(decsum[i])==1:
                if decsum[i][0]==1:
                    sss = getinx(p1_x[i],p2_x[i]-p1_x[i],p1_y[i],p2_y[i]-p1_y[i],p1_z[i],p2_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p1_x[i],p3_x[i]-p1_x[i],p1_y[i],p3_y[i]-p1_y[i],p1_z[i],p3_z[i]-p1_z[i],1./np.tan(dec*D2R))
                elif decsum[i][1]==1:
                    sss = getinx(p1_x[i],p2_x[i]-p1_x[i],p1_y[i],p2_y[i]-p1_y[i],p1_z[i],p2_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p3_x[i],p2_x[i]-p3_x[i],p3_y[i],p2_y[i]-p3_y[i],p3_z[i],p2_z[i]-p3_z[i],1./np.tan(dec*D2R))
                elif decsum[i][2]==1:
                    sss = getinx(p1_x[i],p3_x[i]-p1_x[i],p1_y[i],p3_y[i]-p1_y[i],p1_z[i],p3_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p3_x[i],p2_x[i]-p3_x[i],p3_y[i],p2_y[i]-p3_y[i],p3_z[i],p2_z[i]-p3_z[i],1./np.tan(dec*D2R))
            elif np.sum(decsum[i])==2:
                if decsum[i][0]==0:
                    sss = getinx(p1_x[i],p2_x[i]-p1_x[i],p1_y[i],p2_y[i]-p1_y[i],p1_z[i],p2_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p1_x[i],p3_x[i]-p1_x[i],p1_y[i],p3_y[i]-p1_y[i],p1_z[i],p3_z[i]-p1_z[i],1./np.tan(dec*D2R))
                elif decsum[i][1]==0:
                    sss = getinx(p1_x[i],p2_x[i]-p1_x[i],p1_y[i],p2_y[i]-p1_y[i],p1_z[i],p2_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p3_x[i],p2_x[i]-p3_x[i],p3_y[i],p2_y[i]-p3_y[i],p3_z[i],p2_z[i]-p3_z[i],1./np.tan(dec*D2R))
                elif decsum[i][2]==0:
                    sss = getinx(p1_x[i],p3_x[i]-p1_x[i],p1_y[i],p3_y[i]-p1_y[i],p1_z[i],p3_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p3_x[i],p2_x[i]-p3_x[i],p3_y[i],p2_y[i]-p3_y[i],p3_z[i],p2_z[i]-p3_z[i],1./np.tan(dec*D2R))
            intr[cv].append(np.sqrt(np.sum(np.array(sss)**2.)))
            intr[cv].append(np.sqrt(np.sum(np.array(sst)**2.)))
            intra[cv].append((np.arccos(sss[0]/np.sqrt(sss[0]**2.+sss[1]**2.))*np.sign(sss[1])/D2R)%360)
            intra[cv].append((np.arccos(sst[0]/np.sqrt(sst[0]**2.+sst[1]**2.))*np.sign(sst[1])/D2R)%360)
        return intr,intra

    def getorder(self, xs,ys):
        chains = []
        scut = np.zeros(len(xs),dtype=bool)
        for i in range(len(xs)):
            if len(xs[xs==xs[i]])==1:
                scut[i] = True
            elif len(xs[xs==xs[i]])>2:
                print("0",end='',flush=True)
        dists = []
        pairs = []
        for i in range(len(xs)):
            if scut[i]:
                for j in range(i+1,len(xs)):
                    if scut[j]:
                        dists.append((xs[i]-xs[j])**2.+(ys[i]-ys[j])**2.)
                        pairs.append([i,j])
        pairs = np.array(pairs)[np.argsort(dists)]
        paird = scut
        xs2 = xs.tolist()
        ys2 = ys.tolist()
        cmp = np.arange(len(xs)).tolist()
        for i in range(len(pairs)):
            if paird[pairs[i][0]] and paird[pairs[i][1]]:
                paird[pairs[i][0]] = False
                paird[pairs[i][1]] = False
                xs2.extend([xs[pairs[i][0]],xs[pairs[i][1]]])
                ys2.extend([ys[pairs[i][0]],ys[pairs[i][1]]])
                cmp.extend([pairs[i][0],pairs[i][1]])
        xs2 = np.array(xs2)
        ys2 = np.array(ys2)
        lcut = np.ones(len(xs2),dtype=bool)
        for i in range(len(xs2)):
            if lcut[i]:
                chains.append([])
                chains[-1].append(cmp[i])
                lcut[i] = False
                j = i + 1 - 2*(i%2)
                while xs2[j] != xs2[i]:
                    lcut[j] = False
                    k = np.where(xs2==xs2[j])[0]
                    k = k[k != j][0]
                    chains[-1].append(cmp[k])
                    lcut[k] = False
                    j = k + 1 - 2*(k%2)
                if chains[-1][0] != chains[-1][-1]:
                    chains[-1].append(chains[-1][0])
        return chains

    def convint3(self, intr,intra):
        intx = np.array(intr)*np.cos(np.array(intra)*D2R)
        inty = np.array(intr)*np.sin(np.array(intra)*D2R)
        chkl = []
        ccut = np.ones(len(intr),dtype=bool)
        for i in range(int(len(intr)/2)):
            chkl.append(intx[2*i]+intx[2*i+1])
        chkl = np.array(chkl)
        for i in range(len(chkl)):
            if len(chkl[chkl==chkl[i]])>1:
                ccut[2*i] = False
                ccut[2*i+1] = False
        intx = intx[ccut]
        inty = inty[ccut]
        ocut = self.getorder(intx,inty)
        icut = np.zeros(len(ocut),dtype=bool)
        lens = np.zeros(len(ocut))
        for i in range(len(ocut)):
            for j in range(len(ocut[i])-1):
                lens[i] = lens[i] + np.sqrt((intx[ocut[i][j+1]]-intx[ocut[i][j]])**2.+(inty[ocut[i][j+1]]-inty[ocut[i][j]])**2.)
        mlh = np.amax(lens)
        for i in range(len(ocut)):
            if lens[i]==mlh:
                continue
            o = ocut[i]
            P = np.array([intx[o][0],inty[o][0]])
            for j in range(len(ocut)):
                if j==i:
                    continue
                o1 = ocut[j]
                Ps = np.array([intx[o1],inty[o1]]).T
                if self.isin2(P,Ps):
                    icut[i] = True
                    break
        return [[np.array(intr)[ccut][o].tolist(),np.array(intra)[ccut][o].tolist()] for o in ocut],icut 

    #Plot Zobov Voids
    def pzbv(self,dec,wdth,
    ra0, ra1, cz0, cz1, title, graph = None, zlimits = True, rot = 0, 
    colors = ["blue","blue"],include_gals=True,alpha=0.2, border_alpha = 1,
    horiz_legend_offset=.8, plot_sdss = True, sdss_lim=332.38626, sdss_color='magenta'
             , mag_limit = None, galaxy_point_size=1):
        '''
        Plot Vsquared voids
        '''
        if zlimits:
            cz0 = z_to_comoving_dist(np.array([cz0],dtype=np.float32),.315,1)[0]
            cz1 = z_to_comoving_dist(np.array([cz1],dtype=np.float32),.315,1)[0]
            
        if graph is None:
            fig = plt.figure(1, figsize=(1600/96,800/96))

            ax3, aux_ax3 = setup_axes3(fig, 111, ra0, ra1, cz0, cz1, rot)

            #aux_ax3.set_aspect(1) #Not included in original V2 code?

            plt.title(f"{title} $\delta$ = {dec}$^\circ$", loc='left')
        else:
            fig, ax3, aux_ax3 = graph[0], graph[1], graph[2]
        Intr,Intra = self.trint2(dec)
        #for every unique void ID
        for i in range(np.amax(self.trivids)+1):
            if len(Intr[i])>0:
                #Intr2 = convint(Intr[i])
                #Intra2 = convint(Intra[i])
                #Intr2,Intra2 = convint2(Intr[i],Intra[i])
                Intc2,Icut = self.convint3(Intr[i],Intra[i])
                Intr2 = [Intc[0] for Intc in Intc2]
                Intra2 = [Intc[1] for Intc in Intc2]
                for j in range(len(Intr2)):
                    #if Icut[j]:
                    #    continue
                    
                    color = colors[0] if self.edge[i]==1 else colors[1]
                    aux_ax3.plot(Intra2[j],Intr2[j],alpha=border_alpha,color=color)
                    aux_ax3.fill(Intra2[j],Intr2[j],alpha=alpha,color=color)
                #for j in range(len(Intr2)):
                #    if Icut[j]:
                #        aux_ax3.plot(Intra2[j],Intr2[j],color='blue')
                #        aux_ax3.fill(Intra2[j],Intr2[j],color='white')
        if include_gals:
            gflag_v2, wflag_v2 = self.gflag_v2, self.wflag_v2
            gra, gdec, gr  = self.gra, self.gdec, self.gr
            gdcut = (gr[wflag_v2]*np.sin((gdec[wflag_v2]-dec)*D2R))**2.<wdth**2.
            if mag_limit is not None:
                gdcut = gdcut * self.rmag[wflag_v2] < mag_limit
            aux_ax3.scatter(gra[wflag_v2][gdcut],gr[wflag_v2][gdcut],color='k',s=galaxy_point_size, edgecolor=None)
            gdcut = (gr[gflag_v2]*np.sin((gdec[gflag_v2]-dec)*D2R))**2.<wdth**2.
            if mag_limit is not None:
                gdcut = gdcut * self.rmag[gflag_v2] < mag_limit
            aux_ax3.scatter(gra[gflag_v2][gdcut],gr[gflag_v2][gdcut],color='red',s=galaxy_point_size, edgecolor=None)
            
        if plot_sdss:
            cc=plt.Circle((0, 0), sdss_lim, color=sdss_color,fill=False,linewidth=3)
            ax3.add_artist( cc )
            
        #create legend
        void_legend_handles=[]
        legend_color=clr.to_rgba(colors[1])
        legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
        void_legend_handles.append(
            Line2D([0], [0], label='voids', marker='o', markersize=15, 
                markeredgecolor=colors[1], markerfacecolor=legend_color, linestyle=''
            )
        )
        if colors[0]!=colors[1]:
            legend_color=clr.to_rgba(colors[0])
            legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
            void_legend_handles.append(
                Line2D([0], [0], label='edge voids', marker='o', markersize=15, 
                    markeredgecolor=colors[0], markerfacecolor=legend_color, linestyle='')
            )
        if include_gals:
            void_legend_handles.append(
                Line2D([0], [0], label="wall galaxies", marker='o', markersize=9, 
                    markeredgecolor='k', markerfacecolor='k', linestyle='')
            )
            void_legend_handles.append(
                Line2D([0], [0], label="void galaxies", marker='o', markersize=9, 
                    markeredgecolor='r', markerfacecolor='r', linestyle='')
            )
        if plot_sdss:
            void_legend_handles.append(
                Line2D([0], [0], label='SDSS limit', color=sdss_color, linewidth=5)
            )
            
        aux_ax3.legend(handles=void_legend_handles, loc="upper left", bbox_to_anchor=(horiz_legend_offset,1))
        
        
        self.graph = [fig, ax3, aux_ax3]

        return self.graph


def setup_axes3(fig, rect, ra0, ra1, cz0, cz1, rot):
    """
    Sometimes, things like axis_direction need to be adjusted.
    """

    # rotate a bit for better orientation
    tr_rotate = Affine2D().translate(rot, 0)

    # scale degree to radians
    tr_scale = Affine2D().scale(np.pi/180., 1.)

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorDMS(4)
    tick_formatter1 = FormatterModDMS()

    grid_locator2 = MaxNLocator(3)

    #ra0, ra1 = 195, 236
    #cz0, cz1 = 0., 335. # 306.

    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                        extremes=(ra0, ra1, cz0, cz1),
                                        grid_locator1=grid_locator1,
                                        grid_locator2=grid_locator2,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=None,
                                        )

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # adjust axis
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")

    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")

    ax1.axis["left"].label.set_text(r"r [Mpc h$^{-1}$]")
    ax1.axis["top"].label.set_text(r"$\alpha$")

    #print(ax1.axis["top"].major_ticklabels._grid_info)
    #print(ax1.axis["left"].major_ticklabels)
    #ax1.axis["top"].major_ticklabels.set_visible(False)
    #ax1.axis["left"].major_ticklabels.set_visible(False)
    #ax1.set_xticks([1, 2, 3])
    #ax1.set_yticks([10, 20, 30])
    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.8 # but this has a side effect that the patch is
                        # drawn twice, and possibly over some other
                        # artists. So, we decrease the zorder a bit to
                        # prevent this.
    aux_ax.set_facecolor("white")

    return ax1, aux_ax

class FormatterModDMS(angle_helper.FormatterDMS):


    def __call__(self, direction, factor, values):

        values1 = [v%360 for v in values]
        
        return super().__call__(direction, factor, values1)
