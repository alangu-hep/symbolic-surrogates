import matplotlib.pyplot as plt
import mplhep as mh
plt.style.use(mh.style.CMS)

from collections import defaultdict

class Visualizer:
    def __init__(self, file_prefix, plot_dict, title, combine=False):
        '''
        To Implement:
        1. Logit Distributions
        2. ROC Curves
        3. Number of Parameter Bar Graphs (including vae) OR complexity vs. AUC plots
        4. Latent Traversals
        5. Partial Dependence Plots
        6. Logit vs. Logit Plots
        7. Confusion Matrices
        8. Compression Percentage Bar Charts
        9. Background Rejection Curves
        10. Average Accuracy Tables
        '''

        self.file_prefix = file_prefix
        self.plot_dict = plot_dict
        self.title = title
        self.combine = combine
        self.saveable = defaultdict(list)

        