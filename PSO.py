import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATASET_PATH = "nsl-kdd/KDDTrain+_20Percent.txt"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS_PSO = 5

BASELINE_RESULTS_PATH = "baseline_results.json"
PSO_RESULTS_PATH = "pso_results.json"
COMPARISON_RESULTS_PATH = "comparison_results.json"

CONVERGENCE_PLOT_PATH = "pso_convergence.png"
CONFUSION_MATRIX_PLOT_PATH = "confusion_matrix_comparison.png"
METRICS_COMPARISON_PLOT_PATH = "metrics_comparison.png"

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

N_PARTICLES = 15
N_ITERATIONS = 25
INERTIA_WEIGHT = 0.7
COGNITIVE_COEFF = 1.5
SOCIAL_COEFF = 1.5
VELOCITY_CLAMP_RATIO = 0.2

N_LOG_DISPLAYS = 5

BASELINE_MAX_DEPTH = 3
BASELINE_MAX_LEAF_NODES = 6
BASELINE_MIN_SAMPLES_LEAF = 50

MAX_DEPTH_RANGE = (2, 30)
MIN_SAMPLES_SPLIT_RANGE = (2, 40)
MIN_SAMPLES_LEAF_RANGE = (1, 20)
CRITERION_OPTIONS = ["gini", "entropy"]
CRITERION_RANGE = (0.0, 1.0)

DIMENSION_BOUNDS = [
    MAX_DEPTH_RANGE,
    MIN_SAMPLES_SPLIT_RANGE,
    MIN_SAMPLES_LEAF_RANGE,
    CRITERION_RANGE,
]

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def load_dataset(path):
    df = pd.read_csv(path, names=COLUMN_NAMES)
    print(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def preprocess(df):
    df = df.drop_duplicates()

    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    y = df["label"].apply(lambda v: 0 if str(v).strip().lower() == "normal" else 1)
    X = df.drop(columns=["label"])

    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    print(f"\nSetelah preprocessing: {X.shape[0]} baris, {X.shape[1]} fitur")
    print(f"Distribusi label -> Normal: {(y == 0).sum()} | Serangan: {(y == 1).sum()}")
    return X, y

def clip(value, low, high):
    return max(low, min(high, value))

def decode_solution(position):
    max_depth = int(round(clip(position[0], *MAX_DEPTH_RANGE)))
    min_samples_split = int(round(clip(position[1], *MIN_SAMPLES_SPLIT_RANGE)))
    min_samples_leaf = int(round(clip(position[2], *MIN_SAMPLES_LEAF_RANGE)))
    criterion_value = clip(position[3], *CRITERION_RANGE)
    criterion = CRITERION_OPTIONS[0] if criterion_value < 0.5 else CRITERION_OPTIONS[1]

    return {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
    }

def build_model_from_solution(position):
    params = decode_solution(position)
    return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)

def fitness_function(position, X, y):
    model = build_model_from_solution(position)
    skf = StratifiedKFold(n_splits=CV_FOLDS_PSO, shuffle=True, random_state=RANDOM_STATE)
    try:
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        return scores.mean()
    except Exception:
        return 0.0

def initialize_swarm():
    positions = []
    velocities = []

    for _ in range(N_PARTICLES):
        position = [random.uniform(low, high) for (low, high) in DIMENSION_BOUNDS]
        velocity = [
            random.uniform(-(high - low), (high - low)) * VELOCITY_CLAMP_RATIO
            for (low, high) in DIMENSION_BOUNDS
        ]
        positions.append(position)
        velocities.append(velocity)

    return positions, velocities


def update_velocity(velocity, position, pbest_position, gbest_position):
    new_velocity = []
    for d in range(len(DIMENSION_BOUNDS)):
        r1, r2 = random.random(), random.random()
        cognitive = COGNITIVE_COEFF * r1 * (pbest_position[d] - position[d])
        social = SOCIAL_COEFF * r2 * (gbest_position[d] - position[d])
        v = INERTIA_WEIGHT * velocity[d] + cognitive + social

        low, high = DIMENSION_BOUNDS[d]
        v_max = (high - low) * VELOCITY_CLAMP_RATIO
        v = clip(v, -v_max, v_max)

        new_velocity.append(v)
    return new_velocity

def update_position(position, velocity):
    new_position = []
    for d in range(len(DIMENSION_BOUNDS)):
        low, high = DIMENSION_BOUNDS[d]
        p = clip(position[d] + velocity[d], low, high)
        new_position.append(p)
    return new_position

def run_pso(X, y):
    print("\n===== MENJALANKAN PARTICLE SWARM OPTIMIZATION =====")
    print(f"Jumlah partikel     : {N_PARTICLES}")
    print(f"Jumlah iterasi      : {N_ITERATIONS}")
    print(f"Inertia weight (w)  : {INERTIA_WEIGHT}")
    print(f"Cognitive coeff (c1): {COGNITIVE_COEFF}")
    print(f"Social coeff (c2)   : {SOCIAL_COEFF}")

    positions, velocities = initialize_swarm()

    fitnesses = [fitness_function(pos, X, y) for pos in positions]
    pbest_positions = [pos[:] for pos in positions]
    pbest_fitnesses = fitnesses[:]

    gbest_idx = int(np.argmax(pbest_fitnesses))
    gbest_position = pbest_positions[gbest_idx][:]
    gbest_fitness = pbest_fitnesses[gbest_idx]

    history = []
    log_every = max(1, N_ITERATIONS // N_LOG_DISPLAYS)
    start = time.time()

    for iteration in range(1, N_ITERATIONS + 1):
        for i in range(N_PARTICLES):
            velocities[i] = update_velocity(
                velocities[i], positions[i], pbest_positions[i], gbest_position
            )
            positions[i] = update_position(positions[i], velocities[i])

            fitness = fitness_function(positions[i], X, y)

            if fitness > pbest_fitnesses[i]:
                pbest_fitnesses[i] = fitness
                pbest_positions[i] = positions[i][:]

            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest_position = positions[i][:]

        avg_fitness = float(np.mean(pbest_fitnesses))
        history.append(
            {
                "iteration": iteration,
                "avg_fitness": avg_fitness,
                "best_fitness": float(gbest_fitness),
            }
        )

        if iteration % log_every == 0 or iteration == N_ITERATIONS:
            print(
                f"Iterasi {iteration:4d}/{N_ITERATIONS} | "
                f"Avg Fitness Swarm: {avg_fitness:.4f} | Best Fitness: {gbest_fitness:.4f}"
            )

    duration = time.time() - start
    print(f"\nWaktu optimasi PSO: {duration:.2f} detik")
    print(f"Hyperparameter terbaik: {decode_solution(gbest_position)}")
    print(f"Fitness (CV Accuracy) terbaik: {gbest_fitness:.4f}")

    return gbest_position, gbest_fitness, history, duration


def train_final_model(solution, X_train, y_train):
    start = time.time()
    model = build_model_from_solution(solution)
    model.fit(X_train, y_train)
    duration = time.time() - start
    return model, duration


def evaluate(model, X_test, y_test, duration, label="Decision Tree"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== HASIL EVALUASI: {label} =====")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nKedalaman Tree     : {model.get_depth()}")
    print(f"Jumlah Leaf Nodes  : {model.get_n_leaves()}")

    return {
        "label": label,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "n_features_used": int(X_test.shape[1]),
        "tree_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "training_time_seconds": float(duration),
    }

def plot_pso_convergence(history, filename=CONVERGENCE_PLOT_PATH):
    iterations = [h["iteration"] for h in history]
    avg_fit = [h["avg_fitness"] for h in history]
    best_fit = [h["best_fitness"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, avg_fit, label="Average Fitness Swarm", color="#1565c0",
              linewidth=1.5, linestyle="--")
    plt.plot(iterations, best_fit, label="Best Fitness (gbest)", color="#c62828", linewidth=2)
    plt.xlabel("Iterasi")
    plt.ylabel("Fitness (CV Accuracy)")
    plt.title("Kurva Konvergensi Particle Swarm Optimization")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafik konvergensi PSO disimpan ke '{filename}'")


def plot_confusion_matrices(baseline_cm, pso_cm, filename=CONFUSION_MATRIX_PLOT_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = ["Baseline (Default)", "Setelah Optimasi PSO"]
    matrices = [baseline_cm, pso_cm]
    class_labels = ["Normal", "Serangan"]

    for ax, cm, title in zip(axes, matrices, titles):
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        n = cm_arr.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_labels[:n])
        ax.set_yticklabels(class_labels[:n])
        thresh = cm_arr.max() / 2 if cm_arr.max() > 0 else 0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Confusion Matrix: Decision Tree vs PSO-Optimized")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafik confusion matrix disimpan ke '{filename}'")


def plot_metrics_comparison(baseline_results, pso_results, filename=METRICS_COMPARISON_PLOT_PATH):
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    baseline_vals = [baseline_results.get(m, 0.0) for m in metrics]
    pso_vals = [pso_results.get(m, 0.0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#90a4ae")
    bars2 = ax.bar(x + width / 2, pso_vals, width, label="PSO-Optimized", color="#c62828")

    ax.set_ylabel("Skor")
    ax.set_title("Perbandingan Metrik Evaluasi: Decision Tree vs PSO-Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafik perbandingan metrik disimpan ke '{filename}'")


def compare_results(baseline_results, pso_results):
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    comparison = {"baseline": baseline_results, "pso_optimized": pso_results, "improvement": {}}

    print("\n===== PERBANDINGAN: DECISION TREE vs PSO-OPTIMIZED =====")
    print(f"{'Metrik':<12}{'Decision Tree':>12}{'PSO-Optimized':>16}{'Selisih':>12}{'Selisih (%)':>14}")
    print("-" * 66)
    for m in metrics:
        base_val = baseline_results.get(m, 0.0)
        pso_val = pso_results.get(m, 0.0)
        diff = pso_val - base_val
        pct = (diff / base_val * 100) if base_val != 0 else 0.0
        comparison["improvement"][m] = {"absolute": float(diff), "percentage": float(pct)}
        print(f"{m:<12}{base_val:>12.4f}{pso_val:>16.4f}{diff:>+12.4f}{pct:>+13.2f}%")

    print(
        f"\n{'Tree Depth':<12}{baseline_results.get('tree_depth', 'NA'):>12}"
        f"{pso_results.get('tree_depth', 'NA'):>16}"
    )
    print(
        f"{'N Leaves':<12}{baseline_results.get('n_leaves', 'NA'):>12}"
        f"{pso_results.get('n_leaves', 'NA'):>16}"
    )

    return comparison

def main():
    df = load_dataset(DATASET_PATH)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nData Training : {X_train.shape[0]} baris")
    print(f"Data Testing  : {X_test.shape[0]} baris")

    print("\n===== MELATIH MODEL BASELINE (Decision Tree Default) =====")
    print(f"(max_depth={BASELINE_MAX_DEPTH}, max_leaf_nodes={BASELINE_MAX_LEAF_NODES}, min_samples_leaf={BASELINE_MIN_SAMPLES_LEAF})")
    baseline_model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=BASELINE_MAX_DEPTH,
        max_leaf_nodes=BASELINE_MAX_LEAF_NODES,
        min_samples_leaf=BASELINE_MIN_SAMPLES_LEAF,
    )
    start = time.time()
    baseline_model.fit(X_train, y_train)
    baseline_train_duration = time.time() - start

    baseline_results = evaluate(
        baseline_model, X_test, y_test, baseline_train_duration,
        label=f"Decision Tree Baseline"
    )
    with open(BASELINE_RESULTS_PATH, "w") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"\nHasil evaluasi baseline disimpan ke '{BASELINE_RESULTS_PATH}'")

    best_solution, best_fitness, pso_history, pso_duration = run_pso(X_train, y_train)

    final_model, train_duration = train_final_model(best_solution, X_train, y_train)

    pso_results = evaluate(
        final_model, X_test, y_test, train_duration,
        label="Decision Tree Setelah Optimasi PSO"
    )
    pso_results["best_hyperparameters"] = decode_solution(best_solution)
    pso_results["pso_config"] = {
        "n_particles": N_PARTICLES,
        "n_iterations": N_ITERATIONS,
        "inertia_weight": INERTIA_WEIGHT,
        "cognitive_coeff": COGNITIVE_COEFF,
        "social_coeff": SOCIAL_COEFF,
        "cv_folds_during_pso": CV_FOLDS_PSO,
    }
    pso_results["pso_optimization_time_seconds"] = float(pso_duration)
    pso_results["pso_fitness_history"] = pso_history
    pso_results["best_cv_fitness"] = float(best_fitness)

    with open(PSO_RESULTS_PATH, "w") as f:
        json.dump(pso_results, f, indent=2)
    print(f"\nHasil evaluasi PSO disimpan ke '{PSO_RESULTS_PATH}'")

    plot_pso_convergence(pso_history)

    comparison = compare_results(baseline_results, pso_results)
    with open(COMPARISON_RESULTS_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Hasil perbandingan disimpan ke '{COMPARISON_RESULTS_PATH}'")

    plot_confusion_matrices(baseline_results["confusion_matrix"], pso_results["confusion_matrix"])
    plot_metrics_comparison(baseline_results, pso_results)


if __name__ == "__main__":
    main()