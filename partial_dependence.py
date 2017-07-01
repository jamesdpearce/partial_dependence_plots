from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.append('../utils')
from progress import ProgressBar

def partial_dependence(clf, data, cols,  
                       percentiles = (5, 95), 
                       cluster = True, 
                       n_clusters = 100, 
                       n_points = 100,
                       logit = False,
                       mesh_grid_output = False,
                       show_progress = True,
                       verbose = False):
    
    """Partial dependence of ``cols``.
    
    Partial dependence plots show the marginal effects of one or more features on 
    the model output. 
    
    
    Parameters
    ----------
    
    clf : Sklean classifier class,
        A fitted sklearn classifier object
        
    data : Numpy array of data,
        Numpy array containing columns as features and 
        rows as examples
        
    cols : List of ints, 
        Features to comput partial dependence over. 
    
    percentiles : (max, min) range tuple, default = (5, 95),
        Range for cols to be computed over, given in percentiles
        of distributions. 
        
    cluster : Boolean, default = True, 
        If true data is clustered using K-means. Cluster centers are
        used to compute partial dependence rather than data. This 
        dramatically reduces computation time at little cost in 
        accuracy. 
        
    n_clusters : int, default = 100,
        Number of K-means clusters to compute. 
        
    n_points : int, default = 100, 
        Number of grid points used to calcualte partial dependence per feature. 
        If number of featues is 2 or more this number should be reduced. 
        
    logit : Boolean, default = False, 
        If true perform logit transformation on model outputs. 
        
    mesh_grid_output : Boolean, default = False, 
        If true ouput is given as numpy mesh grid rather than
        flat arrays. This option is useful for 2D and 3D plots.
        
    show_progress : Boolean, default = True, 
        Show progress bar. 
        
    verbose : Boolean, default = False, 
        output level. 
    
    
    
    Returns
    -------
    
    res : List of numpy arrays,
        This list contains the feature values evaluated at along with 
        partial dependence results. res[:-1] contains cols points while res[-1]
        is the partial dependence. By default arrays are 1D however if 
        grid_mesh_output = True the returned list will contain meshgirds.
        
     """
    X = data.copy()
    
    Z_mins = np.percentile( X[:, cols], percentiles[0], axis = 0 )
    Z_maxs = np.percentile( X[:, cols], percentiles[1], axis = 0 )
    
    Z_mesh = np.meshgrid( *[np.linspace(z_jmin, z_jmax, n_points)[:-1] for z_jmin, z_jmax in zip(Z_mins, Z_maxs)] )
    Z = np.stack((z_j.flatten() for z_j in Z_mesh), axis = -1)
    
    if cluster:
        X = KMeans(n_clusters, verbose = int(verbose)).fit(X).cluster_centers_
    
    
    N, N_Z = len(X), len(Z)
    

    pb = ProgressBar()
    p = []
    for i, z_i in enumerate(Z):
        X[:, cols] = np.tile(z_i, (N, 1))

        probs =  clf.predict_proba(X).T[1]
        if logit:
            probs = np.log(probs/(1 - probs))
        p.append( probs.mean(0) )

        if show_progress: pb.update_progress( i/N_Z )
    
    Y = np.array(p) 
    
    if mesh_grid_output:
        Y_res = Y.reshape(Z_mesh[0].shape)
        Z_res = Z_mesh
    else: 
        Z_res = [Z_i.flatten() for Z_i in Z_mesh]
        Y_res = Y
    
    res = Z_res + [Y_res]
    
    return res


def partial_interaction(clf, data, cols,
                        percentiles = (5, 95), 
                        cluster = True, 
                        n_clusters = 100, 
                        n_points = 100,
                        logit = False,
                        mesh_grid_output = True,
                        show_progress = True,
                        verbose = False):
    
    """Partial interactions of ``cols``.
    
    Calculates the partial interactions between cols from their joint 
    and marginal distributions. 
    
    
    Parameters
    ----------
    
    clf : Sklean classifier class,
        A fitted sklearn classifier object
        
    data : Numpy array of data,
        Numpy array containing columns as features and 
        rows as examples
        
    cols : List of ints, 
        Features to comput partial dependence over. 
    
    percentiles : (max, min) range tuple, default = (5, 95),
        Range for cols to be computed over, given in percentiles
        of distributions. 
        
    cluster : Boolean, default = True, 
        If true data is clustered using K-means. Cluster centers are
        used to compute partial dependence rather than data. This 
        dramatically reduces computation time at little cost in 
        accuracy. 
        
    n_clusters : int, default = 100,
        Number of K-means clusters to compute. 
        
    n_points : int, default = 100, 
        Number of grid points used to calcualte partial dependence per feature. 
        If number of featues is 2 or more this number should be reduced. 
        
    logit : Boolean, default = False, 
        If true perform logit transformation on model outputs. 
        
    mesh_grid_output : Boolean, default = False, 
        If true ouput is given as numpy mesh grid rather than
        flat arrays. This option is useful for 2D and 3D plots.
        
    show_progress : Boolean, default = True, 
        Show progress bar. 
        
    verbose : Boolean, default = False, 
        output level. 
    
    
    
    Returns
    -------
    
    Z : List of numpy arrays,
        This list contains the feature values evaluated at along with 
        partial dependence results. res[:-1] contains cols points while res[-1]
        is the partial dependence. By default arrays are 1D however if 
        grid_mesh_output = True the returned list will contain meshgirds.
        
     """
    
    kwargs = { 'percentiles' : percentiles,  
               'n_clusters' : n_clusters, 
               'n_points' : n_points,
               'logit' : logit,
               'show_progress' : show_progress,
               'verbose': verbose }
    X = data.copy()
    
    if cluster:
        X = KMeans(n_clusters, verbose = int(verbose)).fit(X).cluster_centers_
    kwargs['cluster'] = False
    
    Z = partial_dependence(clf, X, cols, mesh_grid_output = True, **kwargs)
    F_joint = Z[-1]

    F_marg = []
    for col_i in cols:
        F_i = partial_dependence(clf, X, [col_i], mesh_grid_output = False, **kwargs)[-1]
        F_marg.append(F_i)
    
    Z[-1] = F_joint - np.meshgrid(*F_marg)[0]
    
    if mesh_grid_output == False:
        Z = [Z_i.flatten() for Z_i in Z]
    
    return Z

def plot_partial_dependences(clf, data, 
                            n_plot_cols = 3, 
                            feature_names = None,
                            cols = None,
                            percentiles = (5, 95), 
                            cluster = True, 
                            n_clusters = 100, 
                            n_points = 100,
                            logit = False,
                            show_progress = True,
                            verbose = False):
    
    """Plots partial dependences of all features.
    
    Plots the single partial dependences of all columsn in X. 
    
    
    Parameters
    ----------
    
    clf : Sklean classifier class,
        A fitted sklearn classifier object
        
    data : Numpy array of data,
        Numpy array containing columns as features and 
        rows as examples
        
    feature_names : List of strings, 
        Column names used in plots. 
        
    cols : List of ints, 
        Features to compute and plot partial dependence over. 
    
    percentiles : (max, min) range tuple, default = (5, 95),
        Range for cols to be computed over, given in percentiles
        of distributions. 
        
    cluster : Boolean, default = True, 
        If true data is clustered using K-means. Cluster centers are
        used to compute partial dependence rather than data. This 
        dramatically reduces computation time at little cost in 
        accuracy. 
        
    n_clusters : int, default = 100,
        Number of K-means clusters to compute. 
        
    n_points : int, default = 100, 
        Number of grid points used to calcualte partial dependence per feature. 
        If number of featues is 2 or more this number should be reduced. 
        
    logit : Boolean, default = False, 
        If true perform logit transformation on model outputs. 
        
    mesh_grid_output : Boolean, default = False, 
        If true ouput is given as numpy mesh grid rather than
        flat arrays. This option is useful for 2D and 3D plots.
        
    show_progress : Boolean, default = True, 
        Show progress bar. 
        
    verbose : Boolean, default = False, 
        output level. 
    
    
    
    Returns
    -------
    
    fig : matplotlib figures
    
    axs : list of matplotlib axes. 
        
     """
    
    
    kwargs = { 'percentiles' : percentiles,  
               'n_clusters' : n_clusters, 
               'n_points' : n_points,
               'logit' : logit,
               'show_progress' : show_progress,
               'verbose': verbose }
    X = data.copy()

    if cols is None:
        cols = range(X.shape[1])

    if feature_names is None:
        feature_names = ['feature ' + str(col) for col in cols]
        
    if cluster:
        X = KMeans(n_clusters, verbose = int(verbose)).fit(X).cluster_centers_
    kwargs['cluster'] = False

    n_plot_cols = min(n_plot_cols, len(cols))
    n_plot_rows = int(np.ceil(len(cols) / float(n_plot_cols)))

    fig = plt.figure()
    axs = []
    for i, (col, feature_name) in enumerate( zip(cols, feature_names) ):
        
        if verbose: print feature_name
            
        ax = fig.add_subplot(n_plot_rows, n_plot_cols, i + 1)
        y, x = partial_dependence(clf, X, cols = [col], mesh_grid_output = False, **kwargs)
        ax.plot(y, x)
        #x.vlines(deciles, [0], 0.05, transform=trans, color='k')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Log odds' if logit else 'Predicted Prob.')
        #ax.set_ylim(ax.get_ylim())
        axs.append(ax)
        
    fig.subplots_adjust(bottom=0.15, top=0.7, left=0.1, right=0.95, wspace=0.4,
                        hspace=0.3)
    return fig, axs
    
    
    