import pandas as pd
import numpy as np
import torch
import ltn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# PART 1: DATA LOADING AND PREPROCESSING

def load_and_preprocess_compas():
    """Load and preprocess COMPAS dataset"""
    
    # Load data
    df = pd.read_csv('datasets/compas-analysis/compas-scores-two-years.csv')
    
    # Select relevant features
    features = [
        'age', 'priors_count', 'juv_fel_count', 'juv_misd_count',
        'c_charge_degree', 'race', 'sex'
    ]
    target = 'two_year_recid'
    
    # Keep only needed columns
    df = df[features + [target]].copy()
    
    df['c_charge_degree'] = df['c_charge_degree'].map({'F': 1, 'M': 0})
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    df['race_binary'] = (df['race'] == 'African-American').astype(int)
    df = df.drop('race', axis=1)
    
    # Remove missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

# PART 2: SIMPLIFIED NEURO-SYMBOLIC MODEL (NO LTN - DIRECT IMPLEMENTATION)

class SimplifiedNeurosymbolicRecidivism(torch.nn.Module):
    """Simplified neuro-symbolic model without complex LTN dependencies"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Neural component
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        
        # Symbolic rule weights (learnable)
        self.rule_weights = torch.nn.Parameter(torch.ones(3) * 0.5)
    
    def is_young(self, x):
        """Fuzzy predicate: young age"""
        age = x[:, 0]  # normalized age
        return torch.sigmoid(-(age - (-1.0)) / 0.5)
    
    def has_many_priors(self, x):
        """Fuzzy predicate: many prior convictions"""
        priors = x[:, 1]  # normalized priors
        return torch.sigmoid((priors - 0.0) / 0.5)
    
    def no_juvenile_record(self, x):
        """Fuzzy predicate: no juvenile felonies"""
        juv_fel = x[:, 2]
        return torch.sigmoid(-(juv_fel + 1.0))
    
    def apply_symbolic_rules(self, x):
        """Apply symbolic rules"""
        rules = torch.zeros(x.shape[0], 3, device=x.device)
        
        # Rule 1: Young with many priors → high risk
        rules[:, 0] = self.is_young(x) * self.has_many_priors(x)
        
        # Rule 2: Older with no juvenile record → low risk
        age = x[:, 0]
        is_older = torch.sigmoid((age - 0.5) / 0.5)
        rules[:, 1] = is_older * self.no_juvenile_record(x)
        
        # Rule 3: Juvenile felonies → moderate risk
        juv_fel = x[:, 2]
        rules[:, 2] = torch.sigmoid(juv_fel)
        
        return rules
    
    def forward(self, x):
        # Neural prediction
        neural_pred = self.neural_net(x).squeeze()
        
        # Symbolic rules
        rules = self.apply_symbolic_rules(x)
        rule_contribution = torch.matmul(rules, torch.nn.functional.softmax(self.rule_weights, dim=0))
        
        alpha = 0.7
        combined = alpha * neural_pred + (1 - alpha) * rule_contribution
        
        return combined  
    
    def explain(self, x):
        """Generate explanation for prediction"""
        with torch.no_grad():
            neural_pred = self.neural_net(x).squeeze()
            rules = self.apply_symbolic_rules(x)
            final_pred = self.forward(x)
            
            return {
                'neural_score': neural_pred.item() if x.shape[0] == 1 else neural_pred.cpu().numpy(),
                'rule_scores': rules.cpu().numpy(),
                'final_prediction': final_pred.item() if x.shape[0] == 1 else final_pred.cpu().numpy()
            }

# PART 3: TRAINING WITH FAIRNESS CONSTRAINTS

def fairness_loss(predictions, race_binary, lambda_fairness=0.1):
    """Compute fairness loss to minimize disparate impact"""
    
    # Separate by race
    protected_group = race_binary == 1  # African-American
    privileged_group = race_binary == 0  # Other
    
    if protected_group.sum() == 0 or privileged_group.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # Calculate positive prediction rates
    protected_rate = predictions[protected_group].mean()
    privileged_rate = predictions[privileged_group].mean()
    
    # Minimize difference (demographic parity)
    fairness_penalty = torch.abs(protected_rate - privileged_rate)
    
    return lambda_fairness * fairness_penalty

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
    """Train the neuro-symbolic model"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    
    # WEIGHTS to handle imbalance
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    criterion = torch.nn.BCELoss()

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values)
    
    # Extract race for fairness
    race_train = torch.FloatTensor(X_train['race_binary'].values)
    race_val = torch.FloatTensor(X_val['race_binary'].values)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_train_tensor)
        
        # Losses
        data_loss = criterion(predictions, y_train_tensor)
        fair_loss = fairness_loss(predictions, race_train, lambda_fairness=0.15)
        
        # Combined loss
        total_loss = data_loss + fair_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor)
                val_pred_binary = (val_predictions > 0.5).float()
                val_accuracy = accuracy_score(y_val_tensor, val_pred_binary)
                val_auc = roc_auc_score(y_val_tensor, val_predictions)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {total_loss.item():.4f} (Data: {data_loss.item():.4f}, Fair: {fair_loss.item():.4f})')
            print(f'  Val Loss: {val_loss.item():.4f}, Acc: {val_accuracy:.4f}, AUC: {val_auc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Load best model
                    model.load_state_dict(torch.load('best_model.pth'))
                    break
    
    return model

# PART 4: EVALUATION AND EXPLAINABILITY

def evaluate_fairness(model, X_test, y_test):
    """Evaluate model fairness metrics"""
    
    X_test_tensor = torch.FloatTensor(X_test.values)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions_binary = (predictions > 0.5).float().numpy()
    
    race = X_test['race_binary'].values
    
    # Calculate metrics for each group
    protected_mask = race == 1
    privileged_mask = race == 0
    
    protected_positive_rate = predictions_binary[protected_mask].mean()
    privileged_positive_rate = predictions_binary[privileged_mask].mean()
    
    disparate_impact = protected_positive_rate / (privileged_positive_rate + 1e-10)
    
    print("\n" + "="*50)
    print("FAIRNESS EVALUATION")
    print("="*50)
    print(f"Protected Group (African-American) Positive Rate: {protected_positive_rate:.3f}")
    print(f"Privileged Group (Other) Positive Rate: {privileged_positive_rate:.3f}")
    print(f"Disparate Impact Ratio: {disparate_impact:.3f}")
    print("  (Ideal: 0.8 - 1.2, closer to 1.0 is more fair)")
    
    # Equal opportunity (TPR parity)
    protected_tpr = predictions_binary[protected_mask & (y_test.values == 1)].mean() if (protected_mask & (y_test.values == 1)).sum() > 0 else 0
    privileged_tpr = predictions_binary[privileged_mask & (y_test.values == 1)].mean() if (privileged_mask & (y_test.values == 1)).sum() > 0 else 0
    
    print(f"\nTrue Positive Rate (Protected): {protected_tpr:.3f}")
    print(f"True Positive Rate (Privileged): {privileged_tpr:.3f}")
    print(f"TPR Difference: {abs(protected_tpr - privileged_tpr):.3f}")
    
    return disparate_impact

def full_evaluation(model, X_test, y_test):
    """Comprehensive model evaluation"""
    
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions_binary = (predictions > 0.5).float().numpy().flatten()
    
    print("\n" + "="*50)
    print("PERFORMANCE EVALUATION")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions_binary, 
                                target_names=['No Recidivism', 'Recidivism']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions_binary))
    
    auc = roc_auc_score(y_test, predictions.numpy())
    print(f"\nAUC-ROC Score: {auc:.4f}")
    
    # Fairness evaluation
    evaluate_fairness(model, X_test, y_test)
    
    # Example explanations
    print("\n" + "="*50)
    print("EXAMPLE EXPLANATIONS")
    print("="*50)
    for i in range(min(3, len(X_test))):
        sample = X_test.iloc[i:i+1]
        sample_tensor = torch.FloatTensor(sample.values)
        exp = model.explain(sample_tensor)
        
        print(f"\nSample {i+1}:")
        print(f"  Neural Score: {exp['neural_score']:.3f}")
        print(f"  Rule Scores: {exp['rule_scores'][0]}")
        print(f"  Final Prediction: {'High Risk' if exp['final_prediction'] > 0.5 else 'Low Risk'} ({exp['final_prediction']:.3f})")

# MAIN EXECUTION

def main():
    print("Loading and preprocessing COMPAS dataset...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_compas()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X_train.columns)}")
    
    # Split training into train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print("\nInitializing Neuro-Symbolic model...")
    model = SimplifiedNeurosymbolicRecidivism(input_dim=X_train.shape[1])
    
    print("\nTraining model with symbolic rules and fairness constraints...")
    model = train_model(model, X_train_split, y_train_split, X_val, y_val, 
                       epochs=1000, lr=0.0001)
    
    print("\nEvaluating on test set...")
    full_evaluation(model, X_test, y_test)
    
    # Save model
    torch.save(model.state_dict(), 'recidivism_neurosymbolic_model.pth')
    print("\nModel saved to 'recidivism_neurosymbolic_model.pth'")

if __name__ == "__main__":
    main()