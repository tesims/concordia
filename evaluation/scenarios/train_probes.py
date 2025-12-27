# MATS Probe Training and Analysis
# Trains linear probes on captured activations
# Includes sanity checks and GM vs Agent comparison
#
# Usage:
#   python train_probes.py --data ./experiment_outputs/run_XXXX/activations.pt

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =============================================================================
# PROBE CLASSES
# =============================================================================

@dataclass
class ProbeResult:
    """Results from training a probe."""
    layer: int
    label_type: str  # "gm" or "agent"
    r2_score: float
    accuracy: float  # For binary classification
    auc: float       # ROC-AUC
    train_r2: float
    test_r2: float
    cross_val_scores: List[float]
    
    def to_dict(self) -> Dict:
        return {
            "layer": int(self.layer),
            "label_type": self.label_type,
            "r2_score": float(self.r2_score),
            "accuracy": float(self.accuracy),
            "auc": float(self.auc),
            "train_r2": float(self.train_r2),
            "test_r2": float(self.test_r2),
            "cross_val_mean": float(np.mean(self.cross_val_scores)) if self.cross_val_scores else 0.0,
            "cross_val_std": float(np.std(self.cross_val_scores)) if self.cross_val_scores else 0.0,
        }


def train_ridge_probe(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 10.0,  # Increased from 1.0 to reduce overfitting
    use_pca: bool = True,  # Reduce dimensions to fight overfitting
    n_components: int = 50,  # Target dimensions (or min of samples/features)
) -> Tuple[Ridge, ProbeResult]:
    """Train a Ridge regression probe."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Optional PCA to reduce overfitting
    if use_pca:
        n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_pca = pca.fit_transform(X)  # For cross-validation
    else:
        X_pca = X

    # Train probe
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)
    
    # Evaluate
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Cross-validation (use PCA-transformed data if applicable)
    cv_scores = cross_val_score(Ridge(alpha=alpha), X_pca, y, cv=5, scoring='r2')
    
    # Binary metrics (threshold at 0.5)
    binary_pred = (test_pred > 0.5).astype(int)
    binary_true = (y_test > 0.5).astype(int)
    
    accuracy = accuracy_score(binary_true, binary_pred)
    
    try:
        auc = roc_auc_score(binary_true, test_pred)
    except ValueError:
        auc = 0.5  # When only one class present
    
    result = ProbeResult(
        layer=-1,  # Set by caller
        label_type="",  # Set by caller
        r2_score=test_r2,
        accuracy=accuracy,
        auc=auc,
        train_r2=train_r2,
        test_r2=test_r2,
        cross_val_scores=cv_scores.tolist(),
    )
    
    return probe, result


def train_mass_mean_probe(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, ProbeResult]:
    """
    Train a mass-mean direction probe (Marks & Tegmark method).
    Often more robust than Ridge for binary concepts.
    """
    
    # Binary labels for direction finding
    binary_y = (y > threshold).astype(bool)
    
    # Compute means
    honest_mean = X[~binary_y].mean(axis=0)
    deceptive_mean = X[binary_y].mean(axis=0)
    
    # Direction vector
    direction = deceptive_mean - honest_mean
    direction = direction / np.linalg.norm(direction)
    
    # Project all points onto direction
    projections = X @ direction
    
    # Evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    test_proj = X_test @ direction
    train_proj = X_train @ direction
    
    # Normalize projections to [0, 1] range
    proj_min, proj_max = projections.min(), projections.max()
    test_pred = (test_proj - proj_min) / (proj_max - proj_min + 1e-8)
    train_pred = (train_proj - proj_min) / (proj_max - proj_min + 1e-8)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Binary metrics
    binary_pred = (test_pred > 0.5).astype(int)
    binary_true = (y_test > 0.5).astype(int)
    
    accuracy = accuracy_score(binary_true, binary_pred)
    
    try:
        auc = roc_auc_score(binary_true, test_pred)
    except ValueError:
        auc = 0.5
    
    result = ProbeResult(
        layer=-1,
        label_type="",
        r2_score=test_r2,
        accuracy=accuracy,
        auc=auc,
        train_r2=train_r2,
        test_r2=test_r2,
        cross_val_scores=[],  # Not applicable for this method
    )
    
    return direction, result


# =============================================================================
# SANITY CHECKS
# =============================================================================

def sanity_check_random_labels(
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 5,
) -> Dict[str, float]:
    """
    Sanity check: probes on shuffled labels should give R² ≈ 0.
    If they don't, probes are memorizing, not learning.
    """
    shuffle_r2s = []
    
    for seed in range(n_shuffles):
        np.random.seed(seed)
        y_shuffled = np.random.permutation(y)
        
        probe, result = train_ridge_probe(X, y_shuffled)
        shuffle_r2s.append(result.r2_score)
    
    return {
        "mean_shuffled_r2": float(np.mean(shuffle_r2s)),
        "std_shuffled_r2": float(np.std(shuffle_r2s)),
        "max_shuffled_r2": float(np.max(shuffle_r2s)),
        "passed": bool(np.mean(shuffle_r2s) < 0.05),  # Should be near 0
    }


def sanity_check_layer_comparison(
    activations: Dict[int, torch.Tensor],
    y: np.ndarray,
) -> Dict[int, float]:
    """
    Compare R² across layers.
    Expect: mid-layers > early layers.
    """
    layer_r2s = {}
    
    for layer, acts in activations.items():
        X = acts.numpy()
        _, result = train_ridge_probe(X, y)
        layer_r2s[layer] = result.r2_score
    
    return layer_r2s


def sanity_check_train_test_gap(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Check for overfitting.
    Large train-test gap = overfit.
    """
    _, result = train_ridge_probe(X, y)
    
    gap = result.train_r2 - result.test_r2
    
    return {
        "train_r2": float(result.train_r2),
        "test_r2": float(result.test_r2),
        "gap": float(gap),
        "passed": bool(gap < 0.2),  # Gap should be small
    }


def sanity_check_label_variance(
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Check that labels have variance.
    If all labels are the same, probes can't learn anything meaningful.
    """
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "n_unique": int(len(np.unique(y))),
        "passed": bool(np.std(y) > 0.1),  # Need some variance
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_analysis(data_path: str) -> Dict[str, Any]:
    """Run complete probe training and analysis."""
    
    print(f"\n{'='*60}")
    print("MATS PROBE TRAINING AND ANALYSIS")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    data = torch.load(data_path)
    
    activations = data["activations"]
    labels = data["labels"]
    config = data.get("config", {})
    
    gm_labels = np.array(labels["gm_labels"])
    agent_labels = np.array(labels["agent_labels"])
    is_deceptive = np.array(labels["is_deceptive"])
    scenarios = labels["scenario"]
    conditions = labels["condition"]
    
    print(f"Loaded {len(gm_labels)} samples")
    print(f"Layers available: {list(activations.keys())}")
    
    results = {
        "sanity_checks": {},
        "layer_analysis": {},
        "gm_vs_agent": {},
        "generalization": {},
        "best_probe": None,
    }
    
    # Choose primary layer (mid-layer)
    layers = sorted(activations.keys())
    mid_layer = layers[len(layers) // 2]
    X_mid = activations[mid_layer].numpy()
    
    print(f"\nPrimary analysis layer: {mid_layer}")
    print(f"Activation shape: {X_mid.shape}")
    
    # ==========================================================================
    # SANITY CHECKS
    # ==========================================================================
    print(f"\n{'='*60}")
    print("SANITY CHECKS")
    print(f"{'='*60}")
    
    # Check 1: Label variance
    print("\n1. Label Variance Check")
    variance_check = sanity_check_label_variance(gm_labels)
    results["sanity_checks"]["label_variance"] = variance_check
    status = "✓ PASSED" if variance_check["passed"] else "✗ FAILED"
    print(f"   GM labels - mean: {variance_check['mean']:.3f}, std: {variance_check['std']:.3f}")
    print(f"   {status}")
    
    # Check 2: Random labels
    print("\n2. Random Labels Check")
    random_check = sanity_check_random_labels(X_mid, gm_labels)
    results["sanity_checks"]["random_labels"] = random_check
    status = "✓ PASSED" if random_check["passed"] else "✗ FAILED"
    print(f"   Shuffled R²: {random_check['mean_shuffled_r2']:.4f} ± {random_check['std_shuffled_r2']:.4f}")
    print(f"   {status} (should be near 0)")
    
    # Check 3: Train-test gap
    print("\n3. Train-Test Gap Check")
    gap_check = sanity_check_train_test_gap(X_mid, gm_labels)
    results["sanity_checks"]["train_test_gap"] = gap_check
    status = "✓ PASSED" if gap_check["passed"] else "✗ FAILED"
    print(f"   Train R²: {gap_check['train_r2']:.3f}, Test R²: {gap_check['test_r2']:.3f}, Gap: {gap_check['gap']:.3f}")
    print(f"   {status}")
    
    # ==========================================================================
    # LAYER COMPARISON
    # ==========================================================================
    print(f"\n{'='*60}")
    print("LAYER ANALYSIS")
    print(f"{'='*60}")
    
    layer_results = {}
    best_layer = None
    best_r2 = -1
    
    for layer in layers:
        X = activations[layer].numpy()
        
        # Train probe on GM labels
        _, gm_result = train_ridge_probe(X, gm_labels)
        gm_result.layer = layer
        gm_result.label_type = "gm"
        
        # Train probe on agent labels
        _, agent_result = train_ridge_probe(X, agent_labels)
        agent_result.layer = layer
        agent_result.label_type = "agent"
        
        layer_results[layer] = {
            "gm": gm_result.to_dict(),
            "agent": agent_result.to_dict(),
        }
        
        print(f"\nLayer {layer}:")
        print(f"  GM labels    - R²: {gm_result.r2_score:.3f}, AUC: {gm_result.auc:.3f}")
        print(f"  Agent labels - R²: {agent_result.r2_score:.3f}, AUC: {agent_result.auc:.3f}")
        
        if gm_result.r2_score > best_r2:
            best_r2 = gm_result.r2_score
            best_layer = layer
    
    results["layer_analysis"] = layer_results
    results["best_probe"] = {"layer": int(best_layer), "r2": float(best_r2)}
    
    print(f"\nBest layer: {best_layer} (R² = {best_r2:.3f})")
    
    # ==========================================================================
    # GM vs AGENT COMPARISON
    # ==========================================================================
    print(f"\n{'='*60}")
    print("GM vs AGENT LABEL COMPARISON")
    print(f"{'='*60}")
    
    # Use best layer
    X_best = activations[best_layer].numpy()
    
    _, gm_result = train_ridge_probe(X_best, gm_labels)
    _, agent_result = train_ridge_probe(X_best, agent_labels)
    
    # Also try mass-mean probe
    gm_direction, gm_mm_result = train_mass_mean_probe(X_best, gm_labels)
    
    results["gm_vs_agent"] = {
        "gm_ridge_r2": float(gm_result.r2_score),
        "agent_ridge_r2": float(agent_result.r2_score),
        "gm_mass_mean_r2": float(gm_mm_result.r2_score),
        "gm_auc": float(gm_result.auc),
        "agent_auc": float(agent_result.auc),
        "gm_wins": bool(gm_result.r2_score > agent_result.r2_score),
    }
    
    print(f"\nGM (Ground Truth):")
    print(f"  Ridge R²:     {gm_result.r2_score:.3f}")
    print(f"  Mass-Mean R²: {gm_mm_result.r2_score:.3f}")
    print(f"  AUC:          {gm_result.auc:.3f}")
    
    print(f"\nAgent (Self-Report):")
    print(f"  Ridge R²:     {agent_result.r2_score:.3f}")
    print(f"  AUC:          {agent_result.auc:.3f}")
    
    if results["gm_vs_agent"]["gm_wins"]:
        print(f"\n✓ GM labels more predictable than agent self-report!")
        print(f"  This suggests agents encode information they don't 'acknowledge'.")
    else:
        print(f"\n  Agent labels equally/more predictable than GM.")
    
    # ==========================================================================
    # GENERALIZATION (Train on 2 scenarios, test on 1)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("GENERALIZATION ANALYSIS")
    print(f"{'='*60}")
    
    unique_scenarios = list(set(scenarios))
    
    if len(unique_scenarios) >= 3:
        generalization_results = {}
        
        for holdout in unique_scenarios:
            # Split by scenario
            train_mask = np.array([s != holdout for s in scenarios])
            test_mask = ~train_mask
            
            X_train = X_best[train_mask]
            X_test = X_best[test_mask]
            y_train = gm_labels[train_mask]
            y_test = gm_labels[test_mask]
            
            # Train and evaluate
            probe = Ridge(alpha=1.0)
            probe.fit(X_train, y_train)
            
            test_pred = probe.predict(X_test)
            test_r2 = r2_score(y_test, test_pred)
            
            generalization_results[holdout] = {
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "test_r2": float(test_r2),
            }
            
            print(f"\nHoldout: {holdout}")
            print(f"  Train on: {[s for s in unique_scenarios if s != holdout]}")
            print(f"  Test R²: {test_r2:.3f}")
        
        avg_generalization = np.mean([r["test_r2"] for r in generalization_results.values()])
        results["generalization"] = {
            "by_scenario": generalization_results,
            "average_r2": float(avg_generalization),
        }
        
        print(f"\nAverage generalization R²: {avg_generalization:.3f}")
    else:
        print("Not enough scenarios for generalization analysis")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n1. Sanity Checks:")
    all_passed = all(
        check.get("passed", True) 
        for check in results["sanity_checks"].values()
    )
    print(f"   {'✓ All passed' if all_passed else '✗ Some failed'}")
    
    print(f"\n2. Best Probe Performance:")
    print(f"   Layer {best_layer}: R² = {best_r2:.3f}")
    
    print(f"\n3. GM vs Agent:")
    if results["gm_vs_agent"]["gm_wins"]:
        print(f"   ✓ GM more predictable (evidence for implicit deception encoding)")
    else:
        print(f"   Agent equally/more predictable")
    
    if "average_r2" in results.get("generalization", {}):
        print(f"\n4. Generalization:")
        print(f"   Average cross-scenario R²: {results['generalization']['average_r2']:.3f}")
    
    return results


def plot_results(results: Dict, output_path: str = None):
    """Generate visualization of results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Layer comparison
    ax1 = axes[0, 0]
    layers = sorted(results["layer_analysis"].keys())
    gm_r2s = [results["layer_analysis"][l]["gm"]["r2_score"] for l in layers]
    agent_r2s = [results["layer_analysis"][l]["agent"]["r2_score"] for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax1.bar(x - width/2, gm_r2s, width, label='GM (Ground Truth)')
    ax1.bar(x + width/2, agent_r2s, width, label='Agent (Self-Report)')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Probe Performance by Layer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2. GM vs Agent comparison
    ax2 = axes[0, 1]
    comparison = results["gm_vs_agent"]
    methods = ['Ridge\n(GM)', 'Ridge\n(Agent)', 'Mass-Mean\n(GM)']
    values = [comparison["gm_ridge_r2"], comparison["agent_ridge_r2"], comparison["gm_mass_mean_r2"]]
    colors = ['green', 'blue', 'darkgreen']
    
    ax2.bar(methods, values, color=colors, alpha=0.7)
    ax2.set_ylabel('R² Score')
    ax2.set_title('GM vs Agent Label Comparison')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Generalization
    ax3 = axes[1, 0]
    if "generalization" in results and "by_scenario" in results["generalization"]:
        gen = results["generalization"]["by_scenario"]
        scenarios = list(gen.keys())
        r2s = [gen[s]["test_r2"] for s in scenarios]
        
        ax3.bar(scenarios, r2s, color='purple', alpha=0.7)
        ax3.axhline(y=results["generalization"]["average_r2"], color='red', 
                   linestyle='--', label=f'Average: {results["generalization"]["average_r2"]:.3f}')
        ax3.set_ylabel('R² Score')
        ax3.set_title('Generalization (Holdout Scenario)')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Not enough scenarios', ha='center', va='center')
        ax3.set_title('Generalization')
    
    # 4. Sanity checks
    ax4 = axes[1, 1]
    checks = results["sanity_checks"]
    check_names = list(checks.keys())
    check_passed = [checks[c].get("passed", True) for c in check_names]
    colors = ['green' if p else 'red' for p in check_passed]
    
    ax4.barh(check_names, [1 if p else 0.5 for p in check_passed], color=colors, alpha=0.7)
    ax4.set_xlim(0, 1.2)
    ax4.set_xlabel('Status')
    ax4.set_title('Sanity Checks')
    
    for i, (name, passed) in enumerate(zip(check_names, check_passed)):
        status = "✓ PASSED" if passed else "✗ FAILED"
        ax4.text(0.1, i, status, va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()
    
    return fig


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train probes on captured activations")
    
    parser.add_argument("--data", type=str, required=True,
                        help="Path to activations.pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save plots")
    
    args = parser.parse_args()
    
    # Run analysis
    results = run_full_analysis(args.data)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        data_path = Path(args.data)
        output_path = data_path.parent / "probe_results.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Generate plots
    if args.plot:
        plot_path = Path(output_path).with_suffix(".png")
        plot_results(results, str(plot_path))


if __name__ == "__main__":
    main()
