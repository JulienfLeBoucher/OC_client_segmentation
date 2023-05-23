import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import r2_score
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram

import xgboost as xgb

import colorcet as cc
  
# NUMERICAL/NUMERICAL Analysis
def display_correlation_matrix(data, features=None, figsize=(11,9)):
    """Display the correlation matrix enlighten with a heatmap

    'features' : enable features restriction.
        When None, display the correlation matrix for all numerical
        columns."""
    sns.set_theme(style="white")
    # Select 'features' or numerical columns
    if features is not None:
        num_data = data.loc[:, features]
    else:
        num_data = data.select_dtypes(include=np.number)

    if num_data.isnull().sum().sum() != 0:
        problem_message = (
            "PROBLEM : there is at least one null value in" 
            + "the data provided"
        )
        print(problem_message)
    else:
        corr = num_data.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=figsize)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr.round(2),
            mask=mask,
            annot=True,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
        plt.title("Correlation Matrix")
        plt.show()
    return None


def numerical_numerical_analysis(
    data,
    ft1_name,
    ft2_name,
    figsize=(5, 5),
    bins=30,
    log_effectives=False,
    log_scale=False,
    precision=3,
):
    if log_effectives:
        x = data[ft1_name]
        y = data[ft2_name]
        xbins = pd.interval_range(start=x.min(), end=x.max())
        ybins = pd.interval_range(start=y.min(), end=y.max())
        X = pd.cut(x, bins=xbins)
        Y = pd.cut(y, bins=ybins)
        mat = pd.crosstab(Y, X)
        mat = mat + 1
        log_mat = np.log(mat)
        log_mat_rev = log_mat.iloc[::-1]  # conventional axis
        sns.heatmap(log_mat_rev, cmap=cmap, square=True)  # To ensure a square pixel
        plt.xticks(
            np.arange(0, len(xbins) + 1, 10), labels=np.arange(xmin, xmax + 1, 10)
        )
        plt.yticks(
            np.arange(0, len(ybins) + 1, 10), labels=np.arange(ymax, ymin - 1, -10)
        )
        plt.ylabel(f"{y.name}")
        plt.xlabel(f"{x.name}")
        plt.title(
            "2D-distribution plot colored with the log of the effectives of each 2D-bin\n"
        )
        plt.show()
    else:
        sns.displot(data=data, x=ft1_name, y=ft2_name, bins=bins, log_scale=log_scale)
        plt.show()
    coeff, _ = pearsonr(data[ft1_name], data[ft2_name])
    print(f"Pearsons' coefficient : {round(coeff , precision)}")
    return None


# CATEGORICAL/NUMERICAL Analysis
def eta_squared(cat_var, num_var, precision=2):
    """Compute and return the squared non-linear correlation
    coefficient equals to SSBetween/SSTotal (ANOVA).

    Parameters :

    - 2 ndarrays without nulls.
    - precision : number of decimals used for rounding"""
    classes_names = cat_var.unique()
    grand_mean = num_var.mean()
    classes = []
    # Compute the effective and the mean of each class name
    for c in classes_names:
        yi = num_var[cat_var == c]
        classes.append({"ni": len(yi), "class_mean": yi.mean()})
        
    SSTotal = sum([(y - grand_mean) ** 2 for y in num_var])
    # Compute the weighted sum of the squared difference between the 
    # class mean and the grand mean. 
    SSBetween = sum([c["ni"] * (c["class_mean"] - grand_mean) ** 2 for c in classes])
    return round(SSBetween / SSTotal, precision)


def categorical_numerical_correlation(
    data,
    cat_name,
    num_name,
    figsize=(15, 8),
    showfliers=True,
    alpha=0.5,
    yticksize=20,
    precision=3,
):
    """Plot a unique figure with multiple boxplot (one per each
    category of the categorical feature). Also compute the squared
    non-linear correlation coefficient."""
    fts = [cat_name, num_name]
    mask = data[cat_name].notnull() & data[num_name].notnull()
    df = data.loc[mask, fts]

    modalities = df[cat_name].unique()
    if data[cat_name].dtype != "category":
        modalities.sort()

    # Generate a list of df's. each df is per modality.
    groups = []
    for m in modalities:
        groups.append(df.loc[data[cat_name] == m, num_name])

    # Define graphical properties.
    medianprops = {"color": "black"}
    meanprops = {
        "marker": "o",
        "markeredgecolor": "black",
        "markerfacecolor": "firebrick",
    }
    flierprops = {"marker": "D", "alpha": alpha, "color": "blue"}

    plt.figure(figsize=figsize)
    plt.boxplot(
        groups,
        labels=modalities,
        showfliers=True,
        medianprops=medianprops,
        vert=False,
        patch_artist=True,
        showmeans=True,
        meanprops=meanprops,
        flierprops=flierprops,
    )
    plt.xticks(size=16)
    plt.yticks(size=yticksize)
    plt.xlabel(num_name, size=16)
    plt.ylabel(cat_name, size=16)
    plt.show()

    cat_var = df[cat_name]
    num_var = df[num_name]
    print(f"eta squared : {eta_squared(cat_var, num_var, precision)}")
    return None


# PCA functions


def display_correlation_circle(
    pca, axis_ranks, features_name=None, label_rotation=0, lims=None, figsize=(7, 7)
):
    """Display the correlation circle in a given factorial plane.

    pca : the sklearn fit pca.
    axis_ranks : t-uple.
        e.g. (0,1) to display the plane of basis (pc1, pc2).
    features_name : list of the features names."""
    pcs = pca.components_
    n_comp = pca.n_components_
    d1, d2 = axis_ranks

    if d2 < n_comp:
        # Initialise the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        # Determine the limits of the chart
        if lims is not None:
            xmin, xmax, ymin, ymax = lims
        elif pcs.shape[1] < 30:
            d = 1.05
            xmin, xmax, ymin, ymax = -d, d, -d, d
        else:
            xmin, xmax, ymin, ymax = (
                min(pcs[d1, :]),
                max(pcs[d1, :]),
                min(pcs[d2, :]),
                max(pcs[d2, :]),
            )
        # Add arrows
        # If there are more than 30 arrows, we do not display the
        # triangle at the end
        if pcs.shape[1] < 30:
            plt.quiver(
                np.zeros(pcs.shape[1]),
                np.zeros(pcs.shape[1]),
                pcs[d1, :],
                pcs[d2, :],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="grey",
            )
        else:
            lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
            ax.add_collection(LineCollection(lines, axes=ax, alpha=0.1, color="black"))
        # Display features names
        if features_name is not None:
            for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                    plt.text(
                        x,
                        y,
                        features_name[i],
                        fontsize="9",
                        ha="center",
                        va="center",
                        rotation=label_rotation,
                        color="blue",
                        alpha=0.75,
                    )
        # Display circle
        circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="b")
        plt.gca().add_artist(circle)
        # Define the limits of the chart
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        # Display grid lines
        plt.plot([-1, 1], [0, 0], color="grey", ls="--")
        plt.plot([0, 0], [-1, 1], color="grey", ls="--")
        # Label the axes, with the percentage of variance explained
        pct_var_d1 = round(100 * pca.explained_variance_ratio_[d1], 1)
        pct_var_d2 = round(100 * pca.explained_variance_ratio_[d2], 1)
        plt.xlabel("PC{} ({}%)".format(d1 + 1, pct_var_d1))
        plt.ylabel("PC{} ({}%)".format(d2 + 1, pct_var_d2))
        # Add title
        plt.title("Correlation Circle (PC{} and PC{})".format(d1 + 1, d2 + 1))
        plt.show(block=False)
    return None


def display_factorial_plane_projection(
    Xt, pca, axis_ranks, labels=None, alpha=1, ax=None,
    illustrative_var=None, figsize=(6, 6), title_size=20,
    palette=None,
):
    """Display the projection of points in a factorial plane.

    Xt : data in the eigenvectors space (orthonormal basis).
    pca : the sklearn fit pca.
    axis_ranks : t-uple.
        e.g. (0,1) to display the plane of base (pc1, pc2).
    labels : list of the labels to be displayed above each point.
    illustrative_var :
    """
    n_comp = pca.n_components_
    d1, d2 = axis_ranks
    
    # Boolean used to plt.show() at the end of the function
    # if ax is not provided
    final_plot = False
    ##

    if d2 < n_comp:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            final_plot = True
        # Display the points
        if illustrative_var is None:
            ax.scatter(Xt[:, d1], Xt[:, d2], alpha=alpha, s=10)
        else:
            illustrative_var = np.array(illustrative_var)
            illustrative_values = np.unique(illustrative_var)
            n_val = len(illustrative_values)
            if palette is None:
                palette = sns.color_palette("tab10", n_val)
            # Plot reversely to see most frequent on top.
            for n, value in enumerate(illustrative_values[::-1]):
                selected = np.where(illustrative_var == value)
                ax.scatter(
                    Xt[selected, d1], Xt[selected, d2],
                    alpha=alpha, label=value, color=palette[n_val - 1 - n],
                    s=10
                )
            ax.legend(bbox_to_anchor=(1.05, 1),
                       loc='upper left',
                       borderaxespad=0.,
                       fontsize=10)

        # Display the labels on the points
        if labels is not None:
            for i, (x, y) in enumerate(Xt[:, [d1, d2]]):
                ax.text(x, y, labels[i], fontsize="14", ha="center", va="center")

        # Define the limits of the chart
        xmin = np.min(Xt[:, [d1]]) * 1.1
        xmax = np.max(Xt[:, [d1]]) * 1.1
        ymin = np.min(Xt[:, [d2]]) * 1.1
        ymax = np.max(Xt[:, [d2]]) * 1.1
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Display grid lines
        ax.plot([-100, 100], [0, 0], color="grey", ls="--")
        ax.plot([0, 0], [-100, 100], color="grey", ls="--")

        # Label the axes, with the percentage of variance explained
        pct_var_d1 = round(100 * pca.explained_variance_ratio_[d1], 1)
        pct_var_d2 = round(100 * pca.explained_variance_ratio_[d2], 1)
        ax.set_xlabel("PC{} ({}%)".format(d1 + 1, pct_var_d1), size=18)
        ax.set_ylabel("PC{} ({}%)".format(d2 + 1, pct_var_d2), size=18)
        # Add title
        ax.set_title(
            f"Projection of points on PC{d1+1} and PC{d2+1}",
            size=title_size
        )
        # ax.show(block=False)
        if final_plot:
            plt.show()
    return None


def display_scree_plot(pca):
    """Display a scree plot for the pca"""

    fig, ax = plt.subplots(figsize=(7, 5))
    scree = pca.explained_variance_ratio_ * 100
    plt.bar(np.arange(len(scree)) + 1, scree, width=0.6)
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker="o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    ax.set_axisbelow(True)
    ax.set_xticks(range(1, len(scree) + 1))
    ax.yaxis.grid(color="gray", linestyle="dashed")
    plt.yticks(np.arange(0, 100 + 1, 10))
    plt.title("Scree plot")
    plt.show(block=False)
    return None


# helper functions
def chunk_list(l, chunk_size):
    return [l[n : n + chunk_size] for n in range(0, len(l), chunk_size)]


# ENCODING
def one_hot_encoding(data, cols, drop=None):
    """Add one_hot encoded columns to the 'data' dataframe.

    cols : must be a list of columns names to be one-hot encoded.
    It is also possible to pass drop in order to avoid multicolinearity.

    Return the fit encoder and the modified dataset (enc, data)."""
    enc = OneHotEncoder(drop=drop)
    # Ensure input shape compatibility
    if len(cols) == 1:
        data_to_encode = np.array(data.loc[:, cols]).reshape(-1, 1)
    else:
        data_to_encode = data.loc[:, cols]
    # Fit, transform and add to the initial dataframe
    encoded_array = enc.fit_transform(data_to_encode).toarray()
    df_encoded = pd.DataFrame(
        encoded_array, index=data.index, columns=enc.get_feature_names_out(cols)
    )
    data = pd.concat([data, df_encoded], axis=1)
    # if initial_drop:
    #     data = data.drop(labels= cols, axis=1)
    return (enc, data)


### Linear regressions
def linearRegressionSummary(model, column_names):
    '''Show a summary of the trained linear regression model'''

    # Plot the coeffients as bars
    fig = plt.figure(figsize=(8, len(column_names)/3))
    fig.suptitle('Linear Regression Coefficients', fontsize=16, y=1.1)
    rects = plt.barh(column_names, model.coef_, color="lightblue")

    # Annotate the bars with the coefficient values
    for rect in rects:
        width = round(rect.get_width(), 2)
        plt.gca().annotate('  {}  '.format(width),
                    xy=(0, rect.get_y()),
                    xytext=(0,2),  
                    textcoords="offset points",  
                    ha='left' if width<0 else 'left', va='bottom')        
    plt.show()


### Model evaluation
def apply_score_func(metric, y_true, y_pred, precision=5):
    """ Compute the score according the 'metric' 
    
    'metric' must be a series of the metrics dataframe I made
    to regroup scoring aliases and scoring function of sklearn.
    
    WARNING, ensure the y's are passed in this order, it is not always
    symmetric!"""
    if metric.kwargs is not None:
        return round(metric.func(y_true, y_pred, **metric.kwargs), precision)
    else:
        return round(metric.func(y_true, y_pred), precision)
    
def score(model, metric, X, y_true, precision=5):
    """Get the model prediction scores using the provided input and
    target features
    
    metric must be in the dataframe metric with columns name and func"""
    y_pred = model.predict(X)
    print(
        f"    {metric.name}",
        apply_score_func(metric, y_true, y_pred)
    )
    return None

def CV_evaluation(
        model, X, y, k=5, scoring="r2", seed=2,
        silent=False, precision=3
    ):
    """Evaluate a model with the k-fold cross validation method."""
    # Create folds
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Compute scores on folds
    results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    
    # Compute mean and std, then round
    mean_score = np.round(results.mean(), precision)
    std_score = np.round(results.std(), precision)
    results = np.round(results, precision)
    
    # Handle cases where loss minimization is processed as 
    # negative maximization
    if scoring.startswith("neg_"):
        mean_score = - mean_score
        results = results * (-1)
        scoring = scoring.lstrip("neg_")  
    
    # Print results    
    if not silent:
        print(f"Cross-validation evaluation : ")
        print("    scores on folds: ", results)
        print("    mean score: ", mean_score)
        print("    std score deviation: ", std_score)
    return (results, mean_score, std_score)


def plot_predictions_vs_real_values(
    y_true, y_pred, show_actual=True, show_residual=True,
):
    """ Plot actual and residuals of predictions vs real values by 
    default. This can be changed through the boolean parameters. """
    if show_actual:
        # displaying predictions vs real values
        _, ax = plt.subplots(figsize=(5, 5))
        _ = PredictionErrorDisplay.from_predictions(
            y_true, y_pred, kind="actual_vs_predicted",
            ax=ax, scatter_kwargs={"alpha": 0.5}
        )
        ax.set_title("Predictions vs real values on the test set")
        plt.show()
    if show_residual:    
        # displaying predictions vs real values
        _, ax = plt.subplots(figsize=(5, 5))
        _ = PredictionErrorDisplay.from_predictions(
            y_true, y_pred, kind="residual_vs_predicted",
            ax=ax, scatter_kwargs={"alpha": 0.5}
        )
        ax.set_title("Residual vs real values on the test set")
        plt.show()
    return None

def xgb_cv(
    alg, X_train, y_train,
    metric='rmse',
    cv_folds=5,
    early_stopping_rounds=10,
    verbose=True
):
    """ Compute CV in order to find the right number of estimators before 
    it over-fits. 
    
    Returns a dataframe with scores' mean and std.
    
    The shape[0] of the dataframe is the optimal number of trees"""
    xgb_param = alg.get_xgb_params()
    xgb_data = xgb.DMatrix(X_train, y_train)
    cv_result = xgb.cv(
        xgb_param,
        xgb_data,
        num_boost_round=alg.get_params()['n_estimators'],
        nfold=cv_folds,
        verbose_eval=verbose,
        metrics=metric,
        early_stopping_rounds=early_stopping_rounds,
    )
    return cv_result

def optimize_estimators_number(
    alg: xgb.Booster,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    metric='rmse',
    cv_folds=5,
    early_stopping_rounds=10,
    verbose=True
):
    """ Compute a cross-validation to find the optimal number of estimators
    to use for a certain booster configuration and set it to that number. 
    
    Return the optimized booster."""
    cv_result = xgb_cv(
        alg, X_train, y_train,
        metric=metric, cv_folds=cv_folds, 
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose
    )
    if verbose:
        display(cv_result.tail(10))

    return alg.set_params(n_estimators=cv_result.shape[0])

### QUERY
def or_query_instruction(feature_name: str, values: list) -> str:
    """ Creates a query instruction when searching for multiple 'values'
    of a 'feature'. """
    query_instruction = ""
    for val in values:
        query_instruction += (feature_name + ' == "' + val + '" | ')
    return query_instruction[:-3]

###
def my_pairplot(data):
    g = sns.PairGrid(data)
    g.map_upper(sns.histplot)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True)
    plt.show()
    
    
### Clustering
pal = sns.color_palette(cc.glasbey, n_colors=25)


def plot_dendrogram(model, **kwargs):
    """ Create linkage matrix and then plot the dendrogram """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return None


def cluster_table_prettier(styler):
    styler.set_caption("The table is to be read by row (feature). "
                       "For a feature, Clusters with higher values"
                       " are displayed with light colors.")
    styler.format(precision=2, thousands=" ")
    # styler.set_table_styles(css)
    styler.background_gradient(axis=1, cmap='viridis')  
    return styler


def add_centroids_display_in_a_plan(centroids, ax, selected_coordinates):
    """
    - Add display of centroids to a matplotlib axis ax.
    
    'selected_coordinates' is a t-uple which indicates the rank of the 
    2 coordinates to extract from the p-coordinates in the p-dim space where
    centroids are expressed.
    """
    nx, ny = selected_coordinates
    centroids_xs = []
    centroids_ys = []
    for k in range(len(centroids)):
        centroids_xs.append(centroids[k][nx]) 
        centroids_ys.append(centroids[k][ny]) 
    return ax.scatter(centroids_xs, centroids_ys, marker='x', s=40, c='k')
    


def display_clusters_in_pca_space_and_tsne_embedding(
    X_proj, pca, X_tsne, label_vec, label_name, palette=pal,
    centroids=None
):
    """ 
    - display the cluster projections on the 2 first factorial planes
    and display the t-SNE embedded clusters.
    - project centroids in pca space and add its display when available.
    
    X_proj and X_tsne are ndarrays of the same length.
    
    """
    if centroids is not None:
        pca_centroids = pca.transform(centroids)
    
    fig, ax = plt.subplots(1,3, figsize=(15,5))

    display_factorial_plane_projection(
        X_proj, pca, (0, 1),
        ax=ax[0],
        title_size=16,
        illustrative_var=label_vec,
        palette=palette
    )
    if centroids is not None:
        add_centroids_display_in_a_plan(
            pca_centroids, ax[0], (0,1)
        )
    
    
    display_factorial_plane_projection(
        X_proj, pca, (1, 2),
        ax=ax[1],
        title_size=16,
        illustrative_var=label_vec,
        palette=palette
    )
    if centroids is not None:
        add_centroids_display_in_a_plan(
            pca_centroids, ax[1], (1,2)
        )

    
    for n, label in enumerate(np.sort(label_vec.unique())):
        sel = np.where(label_vec == label)
        ax[2].scatter(
            X_tsne[sel, 0], X_tsne[sel, 1], 
            s=15, color=palette[n], marker='o', label=label
        )
        
    ax[2].set_title(f't-SNE colored by {label_name}', size=16)
    ax[2].axis('off')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].legend(bbox_to_anchor=(1.05, 0.75))
    fig.tight_layout()
    plt.show()
    return None


def display_clusters_summary_table(X, label_name):
    """ X is a dataframe with all the features and a label_name column. """
    table = (X.groupby(label_name)
            .agg(['mean', 'std', 'min', 'max'])
            .T)
    display(table.style.pipe(cluster_table_prettier))
    return None


def display_features_boxplot_per_cluster(
    X, label_name, layout=(1, 3),
    showfliers=False, palette=pal
):
    """ X is a dataframe with all the features and a label_name column. """
    # fig structure
    n_rows, n_cols = layout
    height = max(3, X[label_name].nunique()/3)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols * 5, n_rows * height))
    # fliers props
    flierprops = dict(marker='x',
                      markersize=8,
                      alpha=0.1,)
    
    for ax, ft in zip(axs.flat, X.columns):
        sns.boxplot(y=label_name, x=ft, data=X,
                    ax=ax, showfliers=showfliers, orient='h',
                    whis=[1,99],
                    flierprops=flierprops,
                    palette=palette,
                    width=0.4)
        ax.xaxis.label.set_size(16)
    plt.suptitle(f'{label_name} : whisker percentiles (1 ; 99)', y=1.01)
    plt.tight_layout()
    plt.show()
    return None
    
    
def display_clusters_effectives_and_percentages(X, label_name, palette=pal):
    """ X is a dataframe with all the features and a label_name column. """
    width = 0.3
    ysize = max(X[label_name].nunique() / 2, 3)
    fig, ax = plt.subplots(figsize=(5, ysize))
    sns.countplot(data=X, y=label_name,
                  palette=palette,
                  ax=ax, width=width)    
    total = len(X)
    xs = []
    ys = []
    pcts = []
    cols = []    
    for p in ax.patches:
        pcts.append(f'{100 * p.get_width() / total:.1f}%\n')
        xs.append(p.get_width())
        ys.append(p.get_y())
        cols.append(p.get_facecolor())
        
    for col, pct, x, y in zip(cols, pcts, xs, ys):    
        ax.annotate(
            pct,
            (x + 0.01 * max(xs) , y),
            color=col, ha='left', va='top', size=10
        )
    ax.set_xlim(0, max(xs) *1.05)
    ax.set_ylabel('Cluster labels', fontsize=12)
    ax.set_xlabel('')
    plt.title("Clusters' effectives", fontsize=18)
    plt.tight_layout()
    plt.show()
    return None
    
    
def display_clusters_comparison(
    X, label_name, layout=(1,3), showfliers=False,
):
    """ For each feature of X :
        1) Plot a table summing the information of clusters.
        2) Plot side-by-side boxplots per cluster for each feature.
        3) Plot effectives and percentages of each clusters.
        
    Args:
        X (pd.DataFrame): Dataframe restricted to the features of
        interest plus the label column.
        label_name (str): the label column name.

    Returns:
        None
    """
    # The summary table 
    display_clusters_summary_table(X, label_name)
    
    # Boxplots  
    display_features_boxplot_per_cluster(X, label_name,
                                         layout=layout, 
                                         showfliers=showfliers)
    
    # The effectives
    display_clusters_effectives_and_percentages(X, label_name)
    return None

### helper functions
def arguments():
        """Returns a tuple containing :
           - a dictionary of the calling function's
           named arguments, and ;
           - a list of calling function's unnamed
           positional arguments.
        """
        from inspect import getargvalues, stack
        posname, kwname, args = getargvalues(stack()[1][0])[-3:]
        posargs = args.pop(posname, [])
        args.update(args.pop(kwname, []))
        return args, posargs
