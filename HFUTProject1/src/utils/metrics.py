from sklearn.metrics import roc_auc_score, average_precision_score

class AdvancedEvaluator:
    def __init__(self):
        self.probs = []
        self.labels = []
    
    def update(self, outputs, labels):
        self.probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
    
    def compute(self):
        return {
            'ROC_AUC': roc_auc_score(self.labels, self.probs),
            'PR_AUC': average_precision_score(self.labels, self.probs),
            'Confidence_Hist': self._plot_hist()
        }
    
    def _plot_hist(self):
        import matplotlib.pyplot as plt
        plt.hist(
            [p for p, l in zip(self.probs, self.labels) if l == 0],
            bins=30, alpha=0.5, label='Real'
        )
        plt.hist(
            [p for p, l in zip(self.probs, self.labels) if l == 1],
            bins=30, alpha=0.5, label='Fake'
        )
        plt.legend()
        return plt.gcf()