import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, 
                           f1_score, mean_squared_error, r2_score, 
                           roc_curve, auc, precision_recall_curve)

def plot_regression_results(y_true, y_pred, title="Résultats de Régression"):
    # Visualise les résultats de régression
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    residuals = y_true - y_pred
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Valeurs prédites')
    axes[0].set_ylabel('Résidus')
    axes[0].set_title('Graphique des résidus')
    axes[0].grid(True)
    
    axes[1].scatter(y_true, y_pred, alpha=0.6)
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1].set_xlabel('Valeurs réelles')
    axes[1].set_ylabel('Valeurs prédites')
    axes[1].set_title('Prédictions vs Réalité')
    axes[1].grid(True)
    
    # Distribution des résidus
    axes[2].hist(residuals, bins=20, alpha=0.7)
    axes[2].set_xlabel('Résidus')
    axes[2].set_ylabel('Fréquence')
    axes[2].set_title('Distribution des résidus')
    axes[2].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_classification_results(y_true, y_pred_proba, y_pred_classes, title="Résultats de Classification"):
    # Visualise les résultats de classification
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred_classes)
    im = axes[0,0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0,0].set_title('Matrice de Confusion')
    plt.colorbar(im, ax=axes[0,0])
    
    tick_marks = np.arange(2)
    axes[0,0].set_xticks(tick_marks)
    axes[0,0].set_xticklabels(['Non-Diabète', 'Diabète'])
    axes[0,0].set_yticks(tick_marks)
    axes[0,0].set_yticklabels(['Non-Diabète', 'Diabète'])
    axes[0,0].set_ylabel('Vraie classe')
    axes[0,0].set_xlabel('Classe prédite')
    
    # Ajouter les valeurs
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        axes[0,0].text(j, i, format(cm[i, j], 'd'),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
    
    # Distribution des probabilités
    axes[0,1].hist(y_pred_proba[y_true == 0], bins=20, alpha=0.7, label='Non-Diabète')
    axes[0,1].hist(y_pred_proba[y_true == 1], bins=20, alpha=0.7, label='Diabète')
    axes[0,1].axvline(0.5, color='black', linestyle='--', label='Seuil')
    axes[0,1].set_xlabel('Probabilité prédite')
    axes[0,1].set_ylabel('Fréquence')
    axes[0,1].set_title('Distribution des Probabilités')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[0,2].plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0,2].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0,2].set_xlabel('Taux de Faux Positifs')
    axes[0,2].set_ylabel('Taux de Vrais Positifs')
    axes[0,2].set_title('Courbe ROC')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[1,0].plot(recall, precision, lw=2)
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Courbe Precision-Recall')
    axes[1,0].grid(True)
    
    # Métriques
    accuracy = accuracy_score(y_true, y_pred_classes)
    recall_val = recall_score(y_true, y_pred_classes)
    f1_val = f1_score(y_true, y_pred_classes)
    
    metrics_text = f"""
    Accuracy: {accuracy:.3f}
    Recall: {recall_val:.3f}
    F1-Score: {f1_val:.3f}
    AUC: {roc_auc:.3f}
    """
    
    axes[1,1].text(0.1, 0.5, metrics_text, fontsize=12, 
                   verticalalignment='center', fontfamily='monospace')
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_title('Métriques de Performance')
    axes[1,1].axis('off')
    
    # Erreurs de classification
    errors = (y_pred_classes != y_true)
    axes[1,2].scatter(range(len(y_true)), y_pred_proba, c=errors, cmap='coolwarm', alpha=0.6)
    axes[1,2].axhline(0.5, color='black', linestyle='--')
    axes[1,2].set_xlabel('Échantillon')
    axes[1,2].set_ylabel('Probabilité prédite')
    axes[1,2].set_title('Erreurs de Classification')
    axes[1,2].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def compare_models(models_results, metric='accuracy'):
    # Compare les résultats de plusieurs modèles
    models = list(models_results.keys())
    scores = [models_results[model][metric] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, alpha=0.7)
    plt.xlabel('Modèles')
    plt.ylabel(metric.capitalize())
    plt.title(f'Comparaison des modèles - {metric.capitalize()}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()