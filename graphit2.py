# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:49:03 2019

@author: s1884344
"""

import matplotlib, matplotlib_venn
import matplotlib.ticker as ticker
import seaborn, math
import wordcloud as wc
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
import matplotlib.cm as cm
plt = matplotlib.pyplot
plt.ioff()

class plot():
    
    def __init__(self,subplot_config=(1,1),figsize=None,projection=None,axes='on',font=None):
        plt.ioff()
        self.plt = plt
        self.figsize = figsize
        self.config = subplot_config
        subplot_kw = {'projection':projection}
        self.fig, self.ax = plt.subplots(subplot_config[1],subplot_config[0],figsize=figsize,subplot_kw=subplot_kw)
        if axes == 'off':
            [axi.set_axis_off() for axi in np.ravel(self.ax)]
        self.plotcount=0
        self.plotcount2 = 0
        if font is not None:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["font.family"] = 'sans-serif'
    
    # Function to return the current target object when multiple subplots are required
    # Saves on line space not having to do this in every method :D
    def get_target_object(self):
        if self.config[0] == 1 and self.config[1] == 1:
            target_object = self.ax
        elif self.config[0] == 1 or self.config[1] == 1:
            target_object = self.ax[self.plotcount]
        else:
            if self.plotcount == self.config[0]:
                self.plotcount =0
                self.plotcount2 +=1
            target_object = self.ax[self.plotcount2][self.plotcount]
        return target_object
    
    
    # Blank plot for illustrations and annotations
    def blank(self,**kwargs):
        target_object = self.get_target_object()
        self.ax_housekeeping(target_object,kwargs)
    
    # Method for a single bar chart or histogram. Data can be a list, series or single column frame
    def single_bar(self,iput,**kwargs):
        # Sort out any local **kwargs
        keys = kwargs.keys()
        options = option_checker(keys)
        [kwargs.pop(i) for i in options]
        fh = format_helper(iput)
        fdict = fh.single_bar(iput,options)
        iput = fdict['d']
        colour, error, edgecolour, linewidth, alpha = [None]*5
        varlist = [colour, error, edgecolour, linewidth, alpha]
        for i,kw in enumerate(['colour', 'error', 'edgecolour', 'linewidth','alpha']):
            if kw in keys:
                varlist[i] = kwargs[kw]
                kwargs.pop(kw)
                    
        labels = None
        if 'labels' in keys:
            labels = kwargs['labels']
            kwargs.pop('labels')
        elif fdict['l'] is not None and 'a' not in fdict.keys():
            labels = fdict['l']
        elif 'a' in fdict.keys():
            if 'annot' not in keys:
                kwargs['annot'] = []
            textsize=10
            if 'xticksize' in keys:
                textsize = kwargs['xticksize']
            offset = [0,0]
            if 'offset' in keys:
                offset = kwargs['offset']
            for i, target in enumerate(zip(fdict['a'],iput)):
                kwargs['annot'].append({'text':target[0],'coords':(i+offset[0],target[1]+offset[1]),'size':textsize})
        if 'cmap' in keys:
            my_cmap = plt.get_cmap(kwargs['cmap'])
            rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
            varlist[0] = my_cmap(rescale(iput))
                    
        self.multi_bar([iput],colour=[varlist[0]],labels=labels,error=[varlist[1]],
                       edgecolour=[varlist[2]],linewidth=[varlist[3]],
                       alpha=[varlist[4]],**kwargs)
        return None
    
    # Method for plotting a multiple-bar chart. Input is a list-of lists or a dataframe
    def multi_bar(self, iput, **kwargs):
        # Convert the input into a list of lists
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.multi_bar(iput,options)
        data = fdict['d']
        
        colour, error, edgecolour, linewidth, alpha = [[None] * len(data)]*5
        varlist = [colour, error, edgecolour, linewidth, alpha]
        for i, kw in enumerate(['colour','error','edgecolour','linewidth','alpha']):
            if kw in keys:
                varlist[i] = kwargs[kw]
        
        labels = None
        if 'labels' in keys:
            labels = kwargs['labels']
        elif fdict['l'] is not None:
            labels = fdict['l']
        
        label_rotation = 0
        if 'label_rotation' in keys:
            label_rotation = kwargs['label_rotation']
            
        width = 0.2
        if 'width' in keys:
            width = kwargs['width']
                    
        offset = 0
        
        position = np.arange(len(data[0]))+offset
        if 'position' in keys:
            position = np.array(kwargs['position'])
            
        bottom = 0
        if 'bottom' in keys:
            bottom = kwargs['bottom']
        target_object = self.get_target_object()
        
        # Additional loop present in this method to plot each different dataset.
        for index, i in enumerate(data):
            if 'stacked' in keys and kwargs['stacked'] is True:
                offset = 0
                if index != 0:
                    bottom = np.array(data[index-1])+bottom
                else:
                    bottom = np.array([0]*len(data[0]))
            if 'horizontal' in keys and kwargs['horizontal'] is True:
                bars = target_object.barh(position,i,color=varlist[0][index],height=width,left=bottom,xerr=varlist[1][index],
                                          edgecolor=varlist[2][index], linewidth=varlist[3][index], alpha=varlist[4][index])
                offset+=width
                if index == len(data)-1:
                    target_object.set_yticks(np.arange(len(data[0]))+(offset-width)/2)
                    if labels is not None:
                        target_object.set_yticklabels(labels)
                    target_object.tick_params(labelrotation = label_rotation,axis='y')
            else:
                bars = target_object.bar(position+offset,i,color=varlist[0][index],width=width,bottom=bottom,yerr=varlist[1][index],
                                         edgecolor=varlist[2][index], linewidth=varlist[3][index], alpha=varlist[4][index])
                offset+=width
                if index == len(data)-1:
                    target_object.set_xticks(np.arange(len(data[0]))+(offset-width)/2)
                    if labels is not None:    
                        target_object.set_xticklabels(labels)
                    target_object.tick_params(labelrotation = label_rotation,axis='x')
            if 'hatch' in keys:
                for bar, pattern in zip(bars,kwargs['hatch'][index]):
                    bar.set_hatch(pattern)
        self.ax_housekeeping(target_object,kwargs)
        
        return None
    
    def single_scatter(self,iput,**kwargs):
        
        # First deal with the input format of the data. default format is a paired list
        # A single series or 2-column frame are also acceptable        
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.single_scatter(iput, options)
        iput = fdict['d']
        
        
        colour, marker, size, alpha, linewidth, edgecolour = [broadcast_up(iput,i,'single_scatter') for i in [None,'o',32,None,None,None]]
        varlist = [colour, marker, size, alpha, linewidth, edgecolour]
        for i, kw in enumerate(['colour','marker','size','alpha','linewidth','edgecolour']):
            if kw in keys:
                varlist[i] = kwargs[kw]
                kwargs.pop(kw)
        
        if 'annotate' in keys:
            kwargs['annot'] = [{'modulate':0.04,'text':i,'coords':(j-0.05,k+0.005),'size':18} for i,j,k in zip(kwargs['annotate'],iput[0],iput[1])]
                                        
        self.multi_scatter([iput],colour=[varlist[0]],marker=[varlist[1]],size=[varlist[2]],
                           alpha=[varlist[3]],linewidths=[varlist[4]],edgecolor=[varlist[5]],**kwargs)
        return None
    
    def density_scatter(self,iput,**kwargs):
        # use the same conversion technique as the single scatter chart.
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.single_scatter(iput, options)
        iput = fdict['d']
        
        cmap, marker, size, alpha, linewidth, edgecolour, bw = ['viridis_r','o',32,None,None,None, None]
        varlist = [cmap, marker, size, alpha, linewidth, edgecolour, bw]
        for i, kw in enumerate(['cmap','marker','size','alpha','linewidth','edgecolour', 'bw']):
            if kw in keys:
                varlist[i] = kwargs[kw]
                kwargs.pop(kw)
         
        xy = np.vstack([list(iput[0]),list(iput[1])])
        z = gaussian_kde(xy,bw_method=varlist[6])(xy)
        idx = z.argsort()
        x, y, z = np.array(iput[0])[idx], np.array(iput[1])[idx], z[idx]
        self.multi_scatter([[x,y]],cmap=[varlist[0]],marker=[varlist[1]],size=[varlist[2]],
                           alpha=[varlist[3]],linewidths=[varlist[4]],edgecolor=[varlist[5]],
                           colour=[z],**kwargs)
        if 'colourbar' in keys:
            plt.colorbar(matplotlib.cm.ScalarMappable(None,varlist[0]),ax=self.get_target_object())
        return None
    
    def density_contour(self,iput,**kwargs):
        # Still the same conversions as the single scatter chart
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.single_scatter(iput, options)
        iput = fdict['d']
        
        cmap, bw, gridsize, shade, lowest = ['viridis_r','scott',100, False, False]
        varlist = [cmap, bw, gridsize, shade, lowest]
        for i, kw in enumerate(['cmap','bw','gridsize','shade','lowest']):
            if kw in keys:
                varlist[i] = kwargs[kw]
                   
        target_object = self.get_target_object()
        seaborn.kdeplot(iput[0],iput[1],ax=target_object,cmap=varlist[0],bw=varlist[1],gridsize=varlist[2], shade=varlist[3],shade_lowest=varlist[4])
        if 'colourbar' in keys:
            plt.colorbar(matplotlib.cm.ScalarMappable(None,varlist[0]),ax=target_object)
        self.ax_housekeeping(target_object,kwargs)
        return None
    
    def histogram(self,iput,**kwargs):
        
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.histogram(iput,options)
        iput = fdict['d']
        
        bins, cumulative, colour = [10, False, None]
        varlist = [bins, cumulative, colour]
        for i, kw in enumerate(['bins','cumulative','colour']):
            if kw in kwargs:
                varlist[i]= kwargs[kw]
                kwargs.pop(kw)
            
        target_object = self.get_target_object()
        target_object.hist(iput, bins=varlist[0], cumulative=varlist[1], color=varlist[2])
        self.ax_housekeeping(target_object,kwargs)
        return None
    
# Function to plot a kernel density estimate plot for univariate data
# output is passed to multi_density1D
    def density1D(self,iput,**kwargs):
        
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.histogram(iput,options)
        iput = fdict['d']
        
        shade, colour, bw, kernel = [False, None, 'scott', 'gau']
        varlist = [shade, colour, bw, kernel]
        for i, kw in enumerate(['shade', 'colour', 'bw', 'kernel']):
            if kw in keys:
                varlist[i] = kwargs[kw]
                kwargs.pop(kw)

        self.multi_density1D([iput], shade=[varlist[0]], colour=[varlist[1]], bw=[varlist[2]], kernel=[varlist[3]])
        return None
    
# Main function for plotting univariate kde plots.
    def multi_density1D(self,iput,**kwargs):
        
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.multi_histogram(iput,options)
        iput = fdict['d']
        shade, colour, bw, kernel = [[False]*len(iput), [None]*len(iput), ['scott']*len(iput), ['gau']*len(iput)]
        varlist = [shade, colour, bw, kernel]
        for i, kw in enumerate(['shade','colour','bw','kernel']):
            if kw in keys:
                varlist[i] = kwargs[kw]
        
        target_object = self.get_target_object()
        for i, data in enumerate(iput):
            seaborn.kdeplot(iput[i], shade=varlist[0][i], color = varlist[1][i],
                            bw = varlist[2][i], kernel= varlist[3][i], ax = target_object)
        self.ax_housekeeping(target_object, kwargs)
        return None
        
    # Since this is a unique function, it makes sense to keep custom annotation within the function instead of the format helper
    def d1_scatter(self,iput,**kwargs):
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.d1_scatter(iput,options)
        iput = fdict['d']
        if 'l' in fdict and fdict['l'] is not None:
            kwargs['xticks'] = fdict['l']
            kwargs['set_xticks'] = list(range(len(fdict['l'])))
        
        if 'jitter' in keys:
            for i, pair in enumerate(iput):
                xcoords = pair[0]
                xcoords = [i+(np.random.randn()*kwargs['jitter']) for i in xcoords]
                iput[i][0] = xcoords
                
        offset = (0,0)
        textsize = 10
        if 'annotate_size' in keys:
            textsize = kwargs['annotate_size']
        if 'annotate_top' in keys and fdict['a'] is not None:
            if 'annot' not in keys:
                kwargs['annot'] = []
            if type(kwargs['annotate_top']) == tuple:
                offset = kwargs['annotate_top']
            cdict = {}
            for i, item in enumerate(iput):
                for x,y in enumerate(item[1]):
                    if x not in cdict.keys():
                        cdict[x]=[(y,i)]
                    else:
                        cdict[x].append((y,i))
            for x in cdict.keys():
                yindex = max(cdict[x],key=lambda x:x[0])
                y = yindex[0] + offset[1]
                x+=offset[0]*len(fdict['a'][yindex[1]])
                kwargs['annot'].append({'text':fdict['a'][yindex[1]],'coords':(x,y),'size':textsize,'rotation':90})
        
        self.multi_scatter(iput,**kwargs)
        pass
    
    # Create a bubble scatter plot to display 3 data dimensions
    def bubble(self,iput,**kwargs):
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.bubble(iput,options)
        iput = fdict['d']
        
        scale = 1.0
        if 'scale' in keys:
            scale = kwargs['scale']
        kwargs['size'] = np.array(iput[2])*scale
        kwargs['size'] = kwargs['size'].tolist()
        iput = iput[:-1]
        self.single_scatter(iput,**kwargs)
        return None
    
    def multi_bubble(self,iput,**kwargs):
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.multi_bubble(iput,options)
        iput = fdict['d']
        
        scale = [1.0]*len(iput)
        if 'scale' in keys:
            scale = kwargs['scale']
        
        kwargs['size'] = [np.array(p[2])*scale[i] for i, p in enumerate(iput)]
        for i, p in enumerate(iput):
            iput[i] = p[:-1]
        self.multi_scatter(iput,**kwargs)
        return None
            
        return None
    
    def multi_scatter(self,iput,**kwargs):
        
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.multi_scatter(iput,options)
        iput = fdict['d']
        
        
        colour, marker, size, alpha, ls, linewidth, edgecolour, cmap = [broadcast_up(iput,i,'multi_scatter') for i in [None,'o',36,None,'-',None,None,None]]
        varlist = [colour, marker, size, alpha, ls, linewidth, edgecolour, cmap]
        for i, kw in enumerate(['colour', 'marker', 'size', 'alpha', 'ls', 'linewidth', 'edgecolour', 'cmap']):
            if kw in keys:
                if type(kwargs[kw]) != list:
                    varlist[i] = [kwargs[kw]]*len(iput)
                else:
                    varlist[i] = kwargs[kw]

        norm = None
        if 'norm' in keys:
            norm = matplotlib.colors. Normalize(vmin=kwargs['norm'][0],vmax=kwargs['norm'][1])
            
        target_object = self.get_target_object()
        for index, i in enumerate(iput):
            target_object.scatter(i[0],i[1],c = varlist[0][index], marker = varlist[1][index],s=varlist[2][index],
                                  alpha=varlist[3][index],norm=norm,linestyle=varlist[4][index], linewidths=varlist[5][index],
                                  edgecolor=varlist[6][index], cmap=varlist[7][index])
            if 'colourbar' in keys:
                plt.colorbar(matplotlib.cm.ScalarMappable(None,varlist[7][index]),ax=target_object)
            if 'fit_line' in keys:
                target_object.plot(np.unique(i[0]),np.poly1d(np.polyfit(i[0],i[1],1))(np.unique(i[0])),color='black')
        self.ax_housekeeping(target_object,kwargs)
        return None
    
    # Function for plotting a line graph input should be a dataframe with the index as the x coordinates
    # A list of lists in the format [[X,y],[X,Y]] is fine as well if the axes aren't the same
    def multi_line(self, iput, **kwargs):
        
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.multi_scatter(iput,options)
        iput = fdict['d']
        
        colour, style, width = [[None]*len(iput),['-']*len(iput),[1]*len(iput)]
        varlist = [colour, style, width]
        for i, kw in enumerate(['colour','style','width']):
            if kw in keys:
                if type(kwargs[kw]) != list:
                    varlist[i] = [kwargs[kw]]*len(iput)
                else:
                    varlist[i] = kwargs[kw]
        
        target_object = self.get_target_object()
        for z, coords in enumerate(iput):
            # smoothing
            if 'smooth' in keys:
                coords_new = np.linspace(min(coords[0]),max(coords[0]),500)
                w = np.isnan(coords[1])
                coords[1] = np.array(coords[1])
                coords[1][w] = 0.
                f = UnivariateSpline(coords[0],coords[1],w=~w)
                #f.set_smoothing_factor(3)
                y_smooth = f(coords_new)
                coords[0], coords[1] = coords_new, y_smooth
            target_object.plot(coords[0],coords[1],color=varlist[0][z],ls=varlist[1][z],linewidth=varlist[2][z])
        self.ax_housekeeping(target_object,kwargs)
        return None
    
    def boxswarm(self, iput, **kwargs):
        keys = kwargs.keys()
        #options = option_checker(keys)
        #fh = format_helper(iput)
        #fdict = fh.multi_bar(iput,options)
        #iput=fdict['d']
        target_object = self.get_target_object()
        seaborn.boxplot(x='method',y='score',data=iput,ax=target_object,saturation=0,fliersize=0)
        seaborn.stripplot(x='method',y='score',data=iput,ax=target_object,hue='bin',jitter=0.25,hue_order=['ClinVar','gnomAD'],
                          palette=['red','C0'])
        self.ax_housekeeping(target_object,kwargs)
        return None
    
    # Function for plotting a ROC curve given values and labels as a list of lists of lists
    def ROC(self, iput, **kwargs):
        from sklearn.metrics import roc_curve, roc_auc_score
        
        keys = kwargs.keys()
        
        if 'colour' in keys:
            kwargs['colour'] = ['grey']+kwargs['colour']
        else:
            kwargs['colour'] = ['grey']+[None]*len(iput)
            
        kwargs['style'] = ['--']+[None]*len(iput)
        kwargs['xlim'] = (0,1.02)
        kwargs['ylim'] = (0,1.02)
        kwargs['xlabel'] = 'False positive rate'
        kwargs['ylabel'] = 'True positive rate'
                
        output = [[[0,1],[0,1]]]
        
        for i in iput:
            fpr, tpr, _ = roc_curve(i[0], i[1])
            auc = roc_auc_score(i[0], i[1])
            
            if 'absoloute' in keys and kwargs['absoloute'] is True:
                if auc < 0.5:
                    fpr, tpr = tpr, fpr
                    
                    # convert to the correct format for the multi-line function
            output.append([fpr,tpr])
        self.multi_line(output, **kwargs)
        return None
    
    def venn(self, iput, **kwargs):
        
        keys = kwargs.keys()
        
        if len(iput) == 3:
            version = 2
        elif len(iput) == 7:
            version = 3
        else:
            return None
        
        if version == 2:
            set_labels=('A','B')
        else:
            set_labels = ('A','B','C')
        if 'labels' in keys:
            set_labels = kwargs['labels']
        
        if self.config != (1,1):
            target_object = self.get_target_object()
            if len(iput) == 3:
                matplotlib_venn.venn2(iput,ax=target_object,set_labels=set_labels)
            else:
                matplotlib_venn.venn3(iput,ax=target_object,set_labels=set_labels)
            self.ax_housekeeping(target_object,kwargs)
        else:
            if len(iput) == 3:
                matplotlib_venn.venn2(iput,set_labels=set_labels)
            else:
                matplotlib_venn.venn3(iput,set_labels=set_labels)
            self.plt_housekeeping(kwargs)
        self.plotcount+=1
        return None
    
    # Input format should be a list, series or single-column dataframe
    def pie(self, iput, **kwargs):
        # Convert the input into a list
        labels = None
        if type(iput) == pd.core.series.Series:
            iput = list(iput)
            labels = list(iput.index)
        if type(iput) == pd.core.frame.DataFrame:
            iput = list(iput[iput.columns[0]])
            labels = list(iput.index)
        
        keys = kwargs.keys()
        
        colour = None
        if 'colour' in keys:
            if type(kwargs['colour']) == list:
                colour = kwargs['colour']
            else:
                colour = len(iput)*[kwargs['colour']]
        
        explode=None
        if 'explode' in keys:
            explode = kwargs['explode']
            
        if 'labels' in keys:
            labels = kwargs['labels']
            
        if 'percent' in keys:
            newlabels = []
            percentages = [round((i/sum(iput))*100,kwargs['percent']) for i in iput]
            for l, p in zip(labels,percentages):
                newlabels.append(l+'\n'+str(p)+'%')
            labels = newlabels
            
        angle = None
        if 'angle' in keys:
            angle = kwargs['angle']
            
        if 'radius' in keys:
            radius = kwargs['radius']
        else:
            radius = 1
        
        wedgeprops = {'edgecolor':'black','linewidth':1}
        target_object = self.get_target_object()
        target_object.pie(iput,explode=explode,colors=colour,labels=labels,startangle=angle,radius=radius,
                          textprops={'fontsize': 22},shadow=False,wedgeprops=wedgeprops)
        self.ax_housekeeping(target_object,kwargs)

        return None
    
    # Input format should be a matrix or a dataframe
    def heatmap(self, iput, **kwargs):
        if type(iput) == pd.core.frame.DataFrame:
            if 'xticks' not in kwargs.keys():
                kwargs['xticks'] = list(iput.columns)
            if 'yticks' not in kwargs.keys():
                kwargs['yticks'] = list(iput.index)
            iput=iput.values
        
        keys = kwargs.keys()
        
        cmap = 'hot'
        if 'cmap' in keys:
            cmap = kwargs['cmap']
            
        if 'xticks' not in keys:
            kwargs['xticks'] = ['']*len(iput[0])
        if 'yticks' not in keys:
            kwargs['yticks'] = ['']*len(iput)
        kwargs['set_xticks'] = np.arange(len(kwargs['xticks']))
        kwargs['set_yticks'] = np.arange(len(kwargs['yticks']))
        if 'annotate' in keys and (kwargs['annotate'] is True or type(kwargs['annotate']) == int):
            xloc = []
            yloc = []
            for i in range(iput.shape[0]):
                for j in range(iput.shape[1]):
                    xloc.append(i)
                    yloc.append(j)
                    
        target_object = self.get_target_object()
        im = target_object.imshow(iput,cmap=cmap)
        if 'colourbar' in keys:
            cbar = plt.colorbar(im, ax=target_object,shrink=0.5)
            if 'cbar_label' in keys:
                cbar.set_label(label=kwargs['cbar_label'],fontsize=24)
            cbar.ax.tick_params(labelsize=18)
        if 'annotate' in keys and (kwargs['annotate'] is True or type(kwargs['annotate']) == int):
            for i, j in zip(xloc, yloc):
                if type(kwargs['annotate']) == int:
                    tx = round(iput[i,j],kwargs['annotate'])
                else:
                    tx = iput[i,j]
                target_object.text(j,i,tx)
        self.ax_housekeeping(target_object,kwargs)
        return None
    
    # Input should be x and y values of a 2d plot. This can also be a series or dataframe
    def hist_heatmap(self,iput,**kwargs):
        
        data = []
        if type(iput) == pd.core.frame.DataFrame:
            data.append(list(iput.index))
            data.append(list(iput[iput.columns[0]]))
        elif type(iput) == pd.core.series.Series:
            data.append(list(iput.index))
            data.append(list(iput))
        else:
            data.append(iput[0])
            data.append(iput[1])
        
        keys = kwargs.keys()
        hexbin = False
        if 'hex' in keys and kwargs['hex'] is True:
            hexbin = True
        
        cmap = 'jet'
        if 'cmap' in keys:
            cmap = kwargs['cmap']
        resolution = (50,50)
        if 'resolution' in keys:
            resolution = kwargs['resolution']
        
        if self.config != (1,1):
            target_object = self.get_target_object()
            if hexbin is False:
                target_object.hist2d(data[0],data[1],resolution,cmap=cmap)
            else:
                target_object.hexbin(data[0],data[1],gridsize=resolution,cmap=cmap)
            self.ax_housekeeping(target_object,kwargs)
        else:
            if hexbin is False:
                plt.hist2d(data[0],data[1],resolution,cmap=cmap)
            else:
                plt.hexbin(data[0],data[1],gridsize=resolution,cmap=cmap)
            self.plt_housekeeping(kwargs)
        self.plotcount+=1
        return None
    
    # Input should be a matrix or a dataframe
    def heatmap_cluster(self, iput, **kwargs):
        # If the input is a dataframe, set the index and columns as tick labels
        keys = kwargs.keys()
        if type(iput) == pd.core.frame.DataFrame:
            if 'xticks' not in keys:
                kwargs['xticks'] = list(iput.columns)
            if 'yticks' not in keys:
                kwargs['yticks'] = list(iput.index)
        mask = None
        if 'mask' in keys:
            mask = iput == (kwargs['mask'])
        cmap = 'RdBu_r'
        if 'cmap' in keys:
            cmap = kwargs['cmap']
        method = 'average'
        if 'method' in keys:
            method = kwargs['method']
        if 'cmap' in keys:
            cmap = kwargs['cmap']
        if self.config != (1,1):
            target_object = self.get_target_object()
            g = seaborn.clustermap(iput,yticklabels=1,xticklabels=1,cmap=cmap,mask=mask,facecolor='black',method=method,ax=target_object)
        else:
            g = seaborn.clustermap(iput,yticklabels=1,xticklabels=1,cmap=cmap,mask=mask,facecolor='black',method=method)
        g.ax_heatmap.set_facecolor('black')
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),fontsize=7)
        #plt.gcf().set_size_inches(self.figsize[0],self.figsize[1])
        #self.plt_housekeeping(kwargs)
        return None
    
    # Draw a boxplot to visually compare two distributions.
    # Input can be a multi-columned dataframe or matrix, alternatively a list of lists.
    def boxplot(self,iput,**kwargs):
        # convert all input formats are accepted by the function, so no need to convert
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.boxplot(iput,options)
        iput = fdict['d']
        
        notch, marker, vert, whis, positions, width, cmap, median_colour, colour = [None, 'o', True, 1.5, None, 0.5, None,'black',None]
        varlist = [notch, marker, vert, whis, positions, width, cmap,median_colour,colour]
        for i, kw in enumerate(['notch','marker','vert','whis','positions','width','cmap','median_colour','colour']):
            if kw in keys:
                varlist[i] = kwargs[kw]
                
        target_object = self.get_target_object()
        ba = target_object.boxplot(iput,patch_artist=True, notch=varlist[0], sym=varlist[1], vert=varlist[2], whis=varlist[3], positions=varlist[4], widths=varlist[5])
        
        if varlist[8] is not None:
            if type(varlist[8]) != list:
                varlist[7] = [varlist[8]]*len(iput)
            patches = ba['boxes']
            for p, c in zip(patches,varlist[8]):
                p.set_facecolor(c)
        if varlist[6] is not None:
            medians = [np.median(i) for i in iput]
            norm = matplotlib.colors.Normalize(vmin=min(medians),vmax=max(medians))
            norm_values = norm(medians)
            colours = varlist[6](norm_values)
            patches = ba['boxes']
            for p, c in zip(patches,colours):
                p.set_facecolor(c)
        for median in ba['medians']:
            median.set_color(varlist[7])
        self.ax_housekeeping(target_object,kwargs)
        return None
    
    def rankplot(self,iput,**kwargs):
        keys = kwargs.keys()
        options = option_checker(keys)
        fh = format_helper(iput)
        fdict = fh.rankplot(iput,options)
        iput = fdict['d']
        
        if 'labels' in keys:
            iput.index = kwargs['labels']
        # replace each column with the rank instead
        for col in iput.columns:
            iput.sort_values(col,ascending=True,inplace=True)
            iput[col] = range(len(iput[col]))
        
        iput['total'] = iput.sum(axis=1)
        iput = iput.sort_values('total',ascending=True)
        iput = iput.drop('total',axis=1)
        # we can achieve the effect we want by stacking a scatter plot and a line chart on top of each other
        # Each row is an independent series for the plot
        if 'labels' not in keys:
            kwargs['ylabels'] = list(iput.index)
        kwargs['set_xticks'] = list(range(len(iput.columns)))
        if 'xticks' not in keys and fdict['l']['cols'] is not None:
            kwargs['xticks'] = fdict['l']['cols']
        kwargs['set_yticks'] = list(range(len(iput.index)))
        if 'yticks' not in kwargs:
            kwargs['yticks'] = [i+1 for i in kwargs['set_yticks']]
            kwargs['yticks'].reverse()
        kwargs['ylim'] = (-0.2,kwargs['set_yticks'][-1]+0.2)
        s_input = [[list(range(len(row)))  , list(row)]   for   i, row in iput.iterrows()]
        self.multi_scatter(s_input,sameplot=True,**kwargs)
        
        self.multi_line(s_input,**kwargs)
    
    def wordcloud(self,iput,**kwargs):
        keys = kwargs.keys()
        
        width, height, horizontal, background, scale = [400, 200, 0.9, 'white', 1]
        varlist = [width,height,horizontal,background, scale]
        for i, kw in enumerate(['width','height','horizontal','background', 'scale']):
            if kw in keys:
                varlist[i] = kwargs[kw]
                
        
        
        w = wc.WordCloud(width=varlist[0],height=varlist[1],prefer_horizontal=varlist[2],background_color=varlist[3],
                         scale=varlist[4]).generate_from_frequencies(iput)
        target_object = self.get_target_object()
        target_object.imshow(w,interpolation='bilinear')
        self.ax_housekeeping(target_object,kwargs)
    
        
    
    def show(self,layout='tight'):
        if layout == 'tight':
            plt.tight_layout()
        plt.show()
        return None
    
    def save(self,filename,layout='tight',format='png', dpi=300):
        if layout == 'tight':
            plt.tight_layout()
        plt.savefig(filename, format=format, dpi=dpi)
        return None
    
    # Method to add common modifiers to a subplot
    def ax_housekeeping(self,target_object,kwargs):
        keys = kwargs.keys()
        if 'sameplot' not in keys:
            self.plotcount+=1
        # Title
        if 'title' in keys:
            s = textsplit(kwargs['title'])
            if len(s) == 2:
                target_object.set_title(s[0],fontsize=s[1])
            else:
                target_object.set_title(s[0])
        # X label
        if 'xlabel' in keys:
            s = textsplit(kwargs['xlabel'])
            if len(s) == 2:
                target_object.set_xlabel(s[0],fontsize=s[1])
            else:
                target_object.set_xlabel(s[0])
        # Y label
        if 'ylabel' in keys:
            s = textsplit(kwargs['ylabel'])
            if len(s) == 2:
                target_object.set_ylabel(s[0],fontsize=s[1])
            else:
                target_object.set_ylabel(s[0])
        # Xlimit
        if 'xlim' in keys:
            target_object.set_xlim(kwargs['xlim'])
        # Ylimit
        if 'ylim' in keys:
            target_object.set_ylim(kwargs['ylim'])
        # Set the position of the ticks
        if 'set_xticks' in keys:
            target_object.set_xticks(kwargs['set_xticks'])
        if 'set_yticks' in keys:
            target_object.set_yticks(kwargs['set_yticks'])
        # Change to custom tick labels
        if 'xticks' in keys:
            target_object.set_xticklabels(kwargs['xticks'])
        if 'yticks' in keys:
            target_object.set_yticklabels(kwargs['yticks'])
        # set line colours
        c = 'black'
        ls = '-'
        if 'linestyle' in keys:
            ls = kwargs['linestyle']
        if 'linecol' in keys:
            c = kwargs['linecol']
        # Add horizontal lines
        if 'hline' in keys:
            if type(kwargs['hline']) != list and type(kwargs['hline']) != tuple:
                target_object.axhline(y=kwargs['hline'],c=c,ls=ls)
            else:
                for i in kwargs['hline']:
                    target_object.axhline(y=i,c=c,ls=ls)
        # Add vertical lines
        if 'vline' in keys:
            if type(kwargs['vline']) != list and type(kwargs['vline']) != tuple:
                target_object.axvline(x=kwargs['vline'],c=c,ls=ls)
            else:
                for i in kwargs['vline']:
                    target_object.axvline(x=i,c=c,ls=ls)
        # fill between two lines on the plot.
        if 'ci_fill' in keys:
            target_object.fill_between(kwargs['ci_fill'][0],kwargs['ci_fill'][1],kwargs['ci_fill'][2],color='grey',alpha=0.2)
        # Calls the specialist functions
        if 'annot' in keys:
            self.annotate(target_object,kwargs['annot'])
        if 'legend_dict' in keys:
            self.custom_legend(target_object,kwargs['legend_dict'])
        if 'axes_dict' in keys:
            self.advanced_axes(target_object,kwargs['axes_dict'])
        if 'rectangle' in keys:
            self.rectangle(target_object,kwargs['rectangle'])
        if 'secondary_axis' in keys:
            kwargs['secondary_axis']['sameplot'] = True
            self.ax_housekeeping(target_object.twinx(),kwargs['secondary_axis'])
        return None
    
    # Method to add custom annotations to any object
    def annotate(self,target_object,instructions):
        # Convert single annotations to a list for easy logic
        if type(instructions) == dict:
            instructions = [instructions]
        # extract all of the information from instructions
        all_coords = []
        # Change the plot order so that annotation plotting proceeds from left to right
        order = [i['coords'][0] for i in instructions]
        # Prevent ties by adding a tiny amount to each value
        for i, j in enumerate(order):
            if len([k for k in order if k == j]) > 1:
                order[i] += (j/1000)+(i/1000)
        instructions = [x for _,x in sorted(zip(order,instructions))]
        for annotation in instructions:
            keys = annotation.keys()
            text = annotation['text']
            s = textsplit(text)
            if len(s) == 2:
                size = s[1]
                text = s[0]
            elif 'size' in keys:
                size = annotation['size']
            else:
                size=None
            xy = annotation['coords']
            xytext = None
            if 'textcoords' in keys:
                xytext = annotation['textcoords']
            if 'rotation' in keys:
                rotation = annotation['rotation']
            else:
                rotation = 0
            if 'bbox' in keys:
                bbox = annotation['bbox']
            else:
                bbox = None
            xycoords = 'data'
            if 'coord_system' in keys:
                xycoords = annotation['coord_system']
            arrowprops = None
            if 'arrow' in keys:
                arrowprops = {'arrowstyle':annotation['arrow']}                
            # Modulate annotations to prevent overlaps
            if 'modulate' in keys:
                threshold = annotation['modulate']
                distances = [math.sqrt(((xy[0]-a[0])**2)+(xy[1]-a[1])**2) for a in all_coords]
                if any([True for i in distances if i < threshold]):
                    counter = 0
                    movement_dict = {0:[2*threshold,0],
                                     1:[2*threshold,-2*threshold],
                                     2:[0,-2*threshold]}
                    all_alts = {}
                    while any([True for i in distances if i < threshold]) and counter <=2:
                        xy_alt = (xy[0]+movement_dict[counter][0],xy[1]+movement_dict[counter][1])
                        distances = [math.sqrt(((xy_alt[0]-a[0])**2)+(xy_alt[1]-a[1])**2) for a in all_coords]
                        all_alts[min(distances)] = xy_alt
                        counter+=1
                    # Pick the best position
                    xy = all_alts[max(all_alts.keys())]
                all_coords.append(xy)
                
        # Add the annotation
            target_object.annotate(text,xy,xytext=xytext,xycoords=xycoords,arrowprops=arrowprops,fontsize=size,rotation=rotation,bbox=bbox)
        return None
    
    # Method to add a custom legend to any object
    def custom_legend(self,target_object,instructions):
        if instructions is None:
            return None
        # import the extra object types we need
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        # extract all of the required data from instructions
        mark_types = instructions['mark_type']
        colours = instructions['colour']
        labels = instructions['label']
        markers = None
        size = 6
        if 'marker' in instructions.keys():
            markers = instructions['marker']
        if 'size' in instructions.keys():
            size = instructions['size']
        loc = 'best'
        if 'location' in instructions.keys():
            loc = instructions['location']
        
        textsize = 10
        if 'textsize' in instructions.keys():
            textsize = instructions['textsize']
        if 'title' in instructions.keys():
            s = textsplit(instructions['title'])
            if len(s) == 1:
                s = [s[0],textsize]
        else:
            s = ['',textsize]
        hatch=None
        if 'hatch' in instructions.keys():
            hatch = instructions['hatch']
        linecol=None
        if 'linecol' in instructions.keys():
            linecol = instructions['linecol']
        
        attributes = [mark_types,colours,labels,markers,size,hatch,linecol]
        
        # make any missing instructions lists of the correct length
        length = max([len(i) for i in attributes if type(i) == list])
        for index, attrib in enumerate(attributes):
            if type(attrib) != list:
                attributes[index] = [attrib]*length
        legend_elements = []
        
        # Assemble our legend objects into a list
        for mark_type, colour, label, marker, size, hatch, linecol in zip(attributes[0],attributes[1],attributes[2],attributes[3],attributes[4],attributes[5],attributes[6]):
            if mark_type == 'line':
                legend_elements.append(Line2D([0],[0],color=colour,label=label,marker=None,linestyle=marker,linewidth=size))
            if mark_type == 'point':
                legend_elements.append(Line2D([0],[0],color=colour,label=label,marker=marker,linestyle='None',ms=size))
            if mark_type == 'block':
                if linecol is None:
                    lc = colour
                else:
                    lc = linecol
                legend_elements.append(Patch(facecolor=colour,edgecolor=lc,label=label,hatch=hatch))
                
        # generate the legend
        legend = target_object.legend(handles=legend_elements, loc = loc, title=s[0], fontsize=textsize)
        legend.get_title().set_fontsize(str(s[1]))
    
    # Merhod to manipulate axes text, label colour and so on
    def advanced_axes(self,target_object,instructions):
        if 'xtop' in instructions.keys():
            target_object.xaxis.tick_top()
            target_object.xaxis.set_label_position('top')
        if 'xtick_rotation' in instructions.keys():
            if target_object == plt:
                target_object.xticks(rotation=instructions['xtick_rotation'],horizontalalignment='center')
            else:
                target_object.set_xticklabels(target_object.get_xticklabels(),rotation=instructions['xtick_rotation'],horizontalalignment='center')
        if 'ytick_rotation' in instructions.keys():
            if target_object == plt:
                target_object.yticks(rotation=instructions['ytick_rotation'])
            else:
                target_object.setyticklabels(target_object.get_yticklabels(),rotation=instructions['ytick_rotation'])
                
        # Axes scales
        if 'xscale' in instructions.keys():
            target_object.set_xscale(instructions['xscale'])
        if 'yscale' in instructions.keys():
            target_object.set_yscale(instructions['yscale'])
                
        # Set axes tick colours
        if 'xtickcolour' in instructions.keys():
            if type(instructions['xtickcolour']) == str:
                tickcolour = [instructions['xtickcolour']]*len(target_object.axes.get_xticklabels())
            else:
                tickcolour = instructions['xtickcolour']
            for i, c in enumerate(tickcolour):
                target_object.axes.get_xticklabels()[i].set_color(c)
                
        if 'ytickcolour' in instructions.keys():
            if type(instructions['ytickcolour']) == str:
                tickcolour = [instructions['ytickcolour']]*len(target_object.axes.get_yticklabels())
            else:
                tickcolour = instructions['ytickcolour']
            for i, c in enumerate(tickcolour):
                target_object.axes.get_yticklabels()[i].set_color(c)
            
        # Set the tick size
        if 'xticksize' in instructions.keys():
            if target_object == plt:
                plt.setp(plt.gca().get_xticklabels(),fontsize=instructions['xticksize'])
            else:
                target_object.tick_params(axis='x',labelsize=instructions['xticksize'])
        if 'yticksize' in instructions.keys():
            if target_object == plt:
                plt.setp(plt.gca().get_yticklabels(),fontsize=instructions['yticksize'])
            else:
                target_object.tick_params(axis='y',labelsize=instructions['yticksize'])
        if 'labelsize' in instructions.keys():
            if target_object == plt:
                plt.rcParams.update({'axes.titlesize':instructions['labelsize']})
            else:
                target_object.yaxis.label.set_size(instructions['labelsize'])
                target_object.xaxis.label.set_size(instructions['labelsize'])
        # Disable axes
        if 'show' in instructions.keys() and instructions['show'] == 'off':
            target_object.axis('off')
        pass
        
    # Instead of each entry being a different rectangle, this works the same way as the legend
    # coords = coordinate tuple
    # size = width/height tuple
    # facecol = face colour
    # linecol = line colour
    def rectangle(self,target_object,instructions):
        coords = instructions['coords']
        size = instructions['size']
        facecol = instructions['facecol']
        linecol = instructions['linecol']
        
        order = 1
        if 'order' in instructions.keys():
            order = instructions['order']
        
        # Convert and strings into lists of the correct length
        attributes = [coords,size,facecol,linecol,order]
        try:
            length = max([len(i) for i in attributes if type(i) == list])
        except:
            length = 1
        for index, attrib in enumerate(attributes):
            if type(attrib) != list:
                attributes[index] = [attrib]*length
        rectangle_objects = []
        
        # Assemble the rectangle objects into a list
        for coords, size, facecol, linecol, order in zip(attributes[0],attributes[1],attributes[2],attributes[3],attributes[4]):
            rectangle_objects.append(plt.Rectangle(coords,size[0],size[1],facecolor=facecol,edgecolor=linecol,zorder=order))
        
        # Plot the objects
        if target_object == plt:
            for rect in rectangle_objects:
                plt.gca().add_patch(rect)
        else:
            for rect in rectangle_objects:
                target_object.add_patch(rect)
        return None
        
    
class format_helper():
    # A class to take complex frames and covert them into the correct input parameters for various graphing functions
    # Could also help generate annotations from point coordinate data
    def __init__(self,data):
        self.tp = self.determine_type(data)
        self.dualsort = lambda x,y,z:list(list(zip(*sorted(zip(x,y))))[z])
    
    # Figure out the type of data we are looking at
    def determine_type(self,data):
        # A cursory examination of data type.
        tp = type(data)
        if tp == pd.core.frame.DataFrame:
            return 'frame'
        # test for a dataframe and column descriptor.
        if len(data) == 2 and type(data[0]) == pd.core.frame.DataFrame:
            return 'frame_desc'
        # Dictionary should be a simple case
        if tp == dict:
            return 'dict'
        if tp == np.ndarray:
            return 'array'
        if tp == pd.core.series.Series:
            return 'series'
        if tp == list:
            return 'list'
    
    def rankplot(self,data,options):
        # Rankplot requires either a frame or a dictionary input
        # labels are extracted from dataframe or dict keys
        l = None
        if self.tp == 'frame':
            d = data
            if 'extract' in options:
                l = {'cols':list(d.columns),'rows':list(d.index)}
        elif self.tp == 'dict':
            d = pd.DataFrame.from_dict(data,orient='index')
            if 'extract' in options:
                l = {'cols':None,'rows':list(d.index)}
        else:
            raise ValueError('This function only supports dataframes and dictionaries')
        return {'d':d,'l':l}
        
    def single_bar(self,data,options):
        # single bar requires a simple list, the default is to return this
        # Should a column reference be provided, return that column from a frame
        # If labels are required, extract and return labels
        # Extract-annotate will label above each bar
        l = None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = list(data)
        elif self.tp == 'series':
            d = list(data)
            if 'extract' in options or 'extract_annotate' in options:
                l = list(data.index)
        elif self.tp == 'dict':
            d = list(data.values())
            if 'extract' in options or 'extract_annotate' in options:
                l = list(data.keys())
        elif self.tp == 'frame_desc':
            d = list(data[0][data[1]])
            if 'extract' in options or 'extract_annotate' in options:
                l = list(data[0].index)
        else:
            raise ValueError('This function only supports lists, arrays, series, dictionaries and descriptive frames')
        if 'sort' in options and l is None:
            d.sort()
        elif 'sort' in options:
            ds = self.dualsort(d,l,0)
            ls = self.dualsort(d,l,1)
            d, l = ds, ls
        if 'reverse' in options:
            d.reverse()
            if l is not None:
                l.reverse()
        if 'extract_annotate' in options:
            return {'d':d,'l':l,'a':l}
        return {'d':d,'l':l}
    
    def multi_bar(self, data, options):
        # multi bar requires a list of lists
        l = None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = data.tolist()
        elif self.tp == 'dict':
            x = pd.DataFrame.from_dict(data).T
            d = [list(x[p]) for p in x.columns]
            if 'extract' in options:
                l = list(x.index)
        elif self.tp == 'frame':
            d = [list(data[p]) for p in data.columns]
            if 'extract' in options:
                l = list(data.index)
        elif self.tp == 'frame_desc':
            d = [list(data[0][p]) for p in data[1]]
            if 'extract' in options:
                l = list(data[0].index)
        else:
            raise ValueError('This function only supports lists, arrays, dictionaries, frames and descriptive frames')
        return {'d':d,'l':l}
    
    def single_scatter(self, data, options):
        # output is a list of lists
        l = None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = data.tolist()
        elif self.tp == 'dict':
            x = list(data.values())
            y = list(data.keys())
            d = [x,y]
        elif self.tp == 'series':
            x = list(data.index)
            y = list(data)
            d = [x,y]
        elif self.tp == 'frame':
            d = [list(data[p]) for p in data.columns]
        elif self.tp == 'frame_desc':
            d = [list(data[0][p]) for p in data[1]]
        else:
            raise ValueError('This function only supports lists, arrays, dictionaries, series, frames and descriptive frames')
        if 'transpose' in options:
            d[0], d[1] = d[1], d[0]
        return {'d':d}
    
    def multi_scatter(self,data,options):
        l = None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = data.tolist()
        elif self.tp == 'dict':
            f = pd.DataFrame.from_dict(data).T
            d = [[list(f.index),f[p]] for p in f.columns]
        elif self.tp == 'frame':
            d = [[list(data.index),data[p]] for p in data.columns]
        elif self.tp == 'frame_desc':
            d = [[list(data[0].index),data[0][p]] for p in data[1]]
        else:
            raise ValueError('This function only supports lists, arrays, dictionaries, frames and descriptive frames')
        if 'transpose' in options:
            for i, seg in enumerate(d):
                d[i] = [seg[1],seg[0]]
        return {'d':d}
    
    def histogram(self,data,options):
        l=None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = data.tolist()
        elif self.tp == 'series':
            d = list(data)
        elif self.tp == 'frame_desc':
            d = list(data[0][data[1]])
        else:
            raise ValueError('This function only supports lists, arrays, series and single column discriptive frames')
        return {'d':d}
    
    def multi_histogram(self, data, options):
        l = None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = data.tolist()
        elif self.tp == 'frame':
            d = [list(data[p]) for p in data.columns]
        elif self.tp == 'frame_desc':
            d = [list(data[0][p]) for p in data[1]]
        else:
            raise ValueError('This function only supports lists, arrays, dataframes and descriptive frames')
        return {'d':d}
    
    def bubble(self, data, options):
        l = None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = data.tolist()
        elif self.tp == 'frame':
            d = [list(data[p]) for p in data.columns]
        elif self.tp == 'frame_desc':
            d = [list(data[0][p]) for p in data[1]]
        else:
            raise ValueError('This function only supports lists and arrays of 3 elements, also frames and descriptive frames of three elements each')
        return {'d':d}
    
    def multi_bubble(self, data, options):
        l = None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = data.tolist()
        elif self.tp == 'frame_desc':
            d = [[list(data[0][p]) for p in q] for q in data[1]]
        else:
            raise ValueError('This function only supports lists and arrays of internal size 3 or descriptive frames with descriptions in the same format')
        return {'d':d}
    
    def d1_scatter(self, data, options):
        l = None
        a = None
        if self.tp == 'list' and 'transpose' in options:
            data = pd.DataFrame(data)
            self.tp = 'frame'
        if self.tp == 'list' or self.tp == 'array':
            xcoords = range(len(data[0]))
            d=[[[x,y] for x,y in zip(xcoords,q)] for q in data]
        elif self.tp == 'frame':
            if 'transpose' in options:
                data=data.T
            if 'extract_annotate' in options:
                a = list(data.index)
            xcoords = range(len(data.columns))
            d = [[[x,y] for x,y in zip(xcoords,q)] for _,q in data.iterrows()]
            if 'extract' in options:
                l = list(data.columns)
        else:
            raise ValueError('This function only supports lists, arrays or dataframes')
        x,y = [],[]
        for item in d:
            x.append([p[0] for p in item])
            y.append([p[1] for p in item])
        return {'d':[[a,b] for a,b in zip(x,y)],'l':l,'a':a}
    
    def boxplot(self, data, options):
        l = None
        if self.tp == 'list':
            d = data
        elif self.tp == 'array':
            d = data.tolist()
        elif self.tp == 'frame':
            d = [list(data[p].dropna()) for p in data.columns]
        elif self.tp == 'frame_desc':
            d = [list(data[0][p]) for p in data[1]]
        else:
            raise ValueError('This function only supports lists, arrays, dataframes and descriptive frames')
        return {'d':d}
    
    
def meanline(xdata, ydata, binsize):
    xmin = min(xdata)
    xmax = max(ydata)
    rang_e = np.arange(xmin, xmax, binsize)
    fdict = {x:[] for x in rang_e}
    for x, y in zip(xdata,ydata):
        for e, i in enumerate(rang_e):
            if e == 0:
                continue
            if x <= i and x>rang_e[e-1]:
                fdict[i].append(y)
    
    xcoords = []
    ycoords = []
    for k, v in fdict.items():
        if len(v) != 0:
            mean = sum(v)/len(v)
        else:
            continue
        xcoords.append(k)
        ycoords.append(mean)
    xcoords = [i-binsize for i in xcoords]
    return xcoords, ycoords

# extract kwargs intended for the format helper
def option_checker(keys):
    output = []
    if 'extract' in keys:
        output.append('extract')
    if 'sort' in keys:
        output.append('sort')
    if 'reverse' in keys:
        output.append('reverse')
    if 'transpose' in keys:
        output.append('transpose')
    if 'extract_annotate' in keys:
        output.append('extract_annotate')
    return output

# Split text entries into text and size entries.
def textsplit(text):
    return text.split('::')

# Function to infer the required shape of arguments and broadcast them up
# To the correct shapes
def broadcast_up(data,attribute,notation='single_scatter'):
    if notation == 'single_bar':
        return attribute
    # single scatter can broadcast to each point or all points
    # This is mostly handled automatically, so no intervention needed
    if notation == 'single_scatter':
        return attribute
    # Multi_scatter is where things start getting weird
    if notation == 'multi_scatter':
        if type(attribute) != list:
            return [attribute]*len(data)
        else:
            return attribute