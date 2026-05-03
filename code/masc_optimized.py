import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')

# ─── 1. DATA LOADING & TEXT PREPROCESSING ─────────────────────────────────────

def load_data():
    data = pd.read_csv('../data/processed/MASC_Features.csv')
    labels = pd.read_csv('../data/processed/Labels.csv')
    y = labels['class']
    data['keywords'].fillna('', inplace=True)
    return data, y

def preprocess_text(text_data):
    stemmer = PorterStemmer()
    preprocessed_text = []
    stop_words = set(stopwords.words('english'))
    for text in text_data:
        text = str(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'url', text)
        text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_.]))@([A-Za-z]+[A-Za-z0-9]+)', '', text)
        text = re.sub(r'[\s]+', ' ', text)
        text = re.sub(r'[-/,\.]', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stop_words])
        stemmed_text = ' '.join([stemmer.stem(word) for word in text.split()])
        preprocessed_text.append(stemmed_text)
    return preprocessed_text

# ─── 2. ABLATION PIPELINE UTILITIES ───────────────────────────────────────────

def get_features(data, scale_numeric=False, ngram_range=(1,1)):
    """Extract features with optional scaling and n-grams."""
    X_text_raw = data['keywords']
    cleaned_text = preprocess_text(X_text_raw)
    
    numeric_cols = [col for col in data.columns if col != 'keywords']
    X_numeric = data[numeric_cols].copy()
    
    # 1. Apply Scaling if requested
    if scale_numeric:
        scaler = StandardScaler()
        X_numeric = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_cols)
        
    # 2. Apply N-Grams if requested
    vect = TfidfVectorizer(ngram_range=ngram_range)
    X_text = vect.fit_transform(cleaned_text)
    
    # Combine (Dense format required for GaussianNB)
    X = pd.concat([X_numeric.reset_index(drop=True), pd.DataFrame(X_text.toarray())], axis=1)
    X.columns = X.columns.astype(str)
    return X

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1': f1_score(y_test, y_pred, average='macro')
    }

# ─── 3. EXPERIMENT RUNNER ─────────────────────────────────────────────────────

def run_ablation_study():
    data, y = load_data()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    # Stratified Split (Match phase 1: 75% train, 10% test, 15% val -> effectively 25% test size for fit)
    # We will just use an 80/20 split for simplicity and better training stability in Phase 2.
    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(len(y_encoded)), y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    # Algorithms to evaluate
    algos = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    # --- PHASE 1: Baseline (Unscaled, Unigram, Default Params) ---
    print("\n--- Running Step 1: Baseline ---")
    X_base = get_features(data, scale_numeric=False, ngram_range=(1,1))
    X_train_b, X_test_b = X_base.iloc[idx_train], X_base.iloc[idx_test]
    results['1_Baseline'] = {}
    for name, model in algos.items():
        results['1_Baseline'][name] = evaluate_model(model, X_train_b, X_test_b, y_train, y_test)
        print(f"Baseline - {name}: {results['1_Baseline'][name]['Accuracy']:.4f}")
        
    # --- PHASE 2: + Feature Scaling ---
    print("\n--- Running Step 2: + Feature Scaling ---")
    X_scaled = get_features(data, scale_numeric=True, ngram_range=(1,1))
    X_train_s, X_test_s = X_scaled.iloc[idx_train], X_scaled.iloc[idx_test]
    results['2_Scaled'] = {}
    for name, model in algos.items():
        results['2_Scaled'][name] = evaluate_model(model, X_train_s, X_test_s, y_train, y_test)
        print(f"Scaled - {name}: {results['2_Scaled'][name]['Accuracy']:.4f}")
        
    # --- PHASE 3: + Bigrams ---
    print("\n--- Running Step 3: + Bigrams ---")
    X_bigram = get_features(data, scale_numeric=True, ngram_range=(1,2))
    X_train_bg, X_test_bg = X_bigram.iloc[idx_train], X_bigram.iloc[idx_test]
    results['3_Bigrams'] = {}
    for name, model in algos.items():
        results['3_Bigrams'][name] = evaluate_model(model, X_train_bg, X_test_bg, y_train, y_test)
        print(f"Bigrams - {name}: {results['3_Bigrams'][name]['Accuracy']:.4f}")
        
    # --- PHASE 4: + Hyperparameter Tuning & Class Balancing ---
    print("\n--- Running Step 4: + Hyperparameter Tuning ---")
    
    # Grids
    grids = {
        'Logistic Regression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42),
            'params': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'params': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
        },
        'MLP': {
            'model': MLPClassifier(max_iter=500, random_state=42),
            'params': {'hidden_layer_sizes': [(100,), (50, 50)], 'alpha': [0.0001, 0.001]}
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {'var_smoothing': [1e-9, 1e-8, 1e-7]}
        }
    }
    
    results['4_Tuned'] = {}
    best_models = {}
    for name, config in grids.items():
        print(f"Tuning {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_bg, y_train)
        best_models[name] = grid.best_estimator_
        
        # Evaluate Best
        y_pred = best_models[name].predict(X_test_bg)
        results['4_Tuned'][name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1': f1_score(y_test, y_pred, average='macro')
        }
        print(f"Tuned - {name}: {results['4_Tuned'][name]['Accuracy']:.4f} (Params: {grid.best_params_})")

    return results, class_names

# ─── 4. VISUALIZATION ─────────────────────────────────────────────────────────

def plot_ablation_study(results):
    os.makedirs('../figures', exist_ok=True)
    
    stages = ['1_Baseline', '2_Scaled', '3_Bigrams', '4_Tuned']
    stage_labels = ['Baseline', '+ Scaling', '+ Bigrams', '+ Tuning']
    algos = list(results['1_Baseline'].keys())
    
    # Extract Accuracy progressions
    accuracy_data = {algo: [results[stage][algo]['Accuracy'] * 100 for stage in stages] for algo in algos}
    
    plt.figure(figsize=(12, 7))
    colors = ['#4A90D9', '#2ECC71', '#F1C40F', '#E74C3C']
    markers = ['o', 's', '^', 'D']
    
    for i, algo in enumerate(algos):
        plt.plot(stage_labels, accuracy_data[algo], marker=markers[i], linewidth=2.5, markersize=8, color=colors[i], label=algo)
        
        # Annotate end points
        plt.text(len(stage_labels)-1 + 0.05, accuracy_data[algo][-1], f"{accuracy_data[algo][-1]:.1f}%", 
                 va='center', fontsize=10, fontweight='bold', color=colors[i])
        
    plt.title('Ablation Study: Impact of Improvisations on Accuracy', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Course Algorithms', fontsize=11, title_fontsize=12)
    plt.ylim(85, 100)
    plt.tight_layout()
    plt.savefig('../figures/ablation_study.png', dpi=200)
    print("\nSaved ablation chart to ../figures/ablation_study.png")

if __name__ == "__main__":
    print("Starting Phase 2 Improvisation Pipeline...")
    results, class_names = run_ablation_study()
    plot_ablation_study(results)
    
    # Save results to a CSV for easy table creation in README
    records = []
    for stage, algos in results.items():
        for algo, metrics in algos.items():
            records.append({
                'Stage': stage,
                'Algorithm': algo,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1']
            })
    pd.DataFrame(records).to_csv('../figures/ablation_results.csv', index=False)
    print("Saved metrics to ../figures/ablation_results.csv")
