"""
This type stub file was generated by pyright.
"""

class DecisionBoundaryDisplay:
    """Decisions boundary visualization.

    It is recommended to use
    :func:`~sklearn.inspection.DecisionBoundaryDisplay.from_estimator`
    to create a :class:`DecisionBoundaryDisplay`. All parameters are stored as
    attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    xx0 : ndarray of shape (grid_resolution, grid_resolution)
        First output of :func:`meshgrid <numpy.meshgrid>`.

    xx1 : ndarray of shape (grid_resolution, grid_resolution)
        Second output of :func:`meshgrid <numpy.meshgrid>`.

    response : ndarray of shape (grid_resolution, grid_resolution)
        Values of the response function.

    xlabel : str, default=None
        Default label to place on x axis.

    ylabel : str, default=None
        Default label to place on y axis.

    Attributes
    ----------
    surface_ : matplotlib `QuadContourSet` or `QuadMesh`
        If `plot_method` is 'contour' or 'contourf', `surface_` is a
        :class:`QuadContourSet <matplotlib.contour.QuadContourSet>`. If
        `plot_method is `pcolormesh`, `surface_` is a
        :class:`QuadMesh <matplotlib.collections.QuadMesh>`.

    ax_ : matplotlib Axes
        Axes with confusion matrix.

    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, *, xx0, xx1, response, xlabel=..., ylabel=...) -> None:
        ...
    
    def plot(self, plot_method=..., ax=..., xlabel=..., ylabel=..., **kwargs): # -> Self@DecisionBoundaryDisplay:
        """Plot visualization.

        Parameters
        ----------
        plot_method : {'contourf', 'contour', 'pcolormesh'}, default='contourf'
            Plotting method to call when plotting the response. Please refer
            to the following matplotlib documentation for details:
            :func:`contourf <matplotlib.pyplot.contourf>`,
            :func:`contour <matplotlib.pyplot.contour>`,
            :func:`pcolomesh <matplotlib.pyplot.pcolomesh>`.

        ax : Matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        xlabel : str, default=None
            Overwrite the x-axis label.

        ylabel : str, default=None
            Overwrite the y-axis label.

        **kwargs : dict
            Additional keyword arguments to be passed to the `plot_method`.

        Returns
        -------
        display: :class:`~sklearn.inspection.DecisionBoundaryDisplay`
        """
        ...
    
    @classmethod
    def from_estimator(cls, estimator, X, *, grid_resolution=..., eps=..., plot_method=..., response_method=..., xlabel=..., ylabel=..., ax=..., **kwargs): # -> DecisionBoundaryDisplay:
        """Plot decision boundary given an estimator.

        Read more in the :ref:`User Guide <visualizations>`.

        Parameters
        ----------
        estimator : object
            Trained estimator used to plot the decision boundary.

        X : {array-like, sparse matrix, dataframe} of shape (n_samples, 2)
            Input data that should be only 2-dimensional.

        grid_resolution : int, default=100
            Number of grid points to use for plotting decision boundary.
            Higher values will make the plot look nicer but be slower to
            render.

        eps : float, default=1.0
            Extends the minimum and maximum values of X for evaluating the
            response function.

        plot_method : {'contourf', 'contour', 'pcolormesh'}, default='contourf'
            Plotting method to call when plotting the response. Please refer
            to the following matplotlib documentation for details:
            :func:`contourf <matplotlib.pyplot.contourf>`,
            :func:`contour <matplotlib.pyplot.contour>`,
            :func:`pcolomesh <matplotlib.pyplot.pcolomesh>`.

        response_method : {'auto', 'predict_proba', 'decision_function', \
                'predict'}, default='auto'
            Specifies whether to use :term:`predict_proba`,
            :term:`decision_function`, :term:`predict` as the target response.
            If set to 'auto', the response method is tried in the following order:
            :term:`decision_function`, :term:`predict_proba`, :term:`predict`.
            For multiclass problems, :term:`predict` is selected when
            `response_method="auto"`.

        xlabel : str, default=None
            The label used for the x-axis. If `None`, an attempt is made to
            extract a label from `X` if it is a dataframe, otherwise an empty
            string is used.

        ylabel : str, default=None
            The label used for the y-axis. If `None`, an attempt is made to
            extract a label from `X` if it is a dataframe, otherwise an empty
            string is used.

        ax : Matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Additional keyword arguments to be passed to the
            `plot_method`.

        Returns
        -------
        display : :class:`~sklearn.inspection.DecisionBoundaryDisplay`
            Object that stores the result.

        See Also
        --------
        DecisionBoundaryDisplay : Decision boundary visualization.
        ConfusionMatrixDisplay.from_estimator : Plot the confusion matrix
            given an estimator, the data, and the label.
        ConfusionMatrixDisplay.from_predictions : Plot the confusion matrix
            given the true and predicted labels.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.inspection import DecisionBoundaryDisplay
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]
        >>> classifier = LogisticRegression().fit(X, iris.target)
        >>> disp = DecisionBoundaryDisplay.from_estimator(
        ...     classifier, X, response_method="predict",
        ...     xlabel=iris.feature_names[0], ylabel=iris.feature_names[1],
        ...     alpha=0.5,
        ... )
        >>> disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")
        <...>
        >>> plt.show()
        """
        ...
    

