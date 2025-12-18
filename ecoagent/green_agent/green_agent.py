from pathlib import Path
import pandas as pd, yaml

class DatasetBudget:
    def __init__(self, max_datasets:int):
        self.max = int(max_datasets)
        self._used = set()
    def request(self, ds_id:str):
        if ds_id in self._used: return True
        if len(self._used) >= self.max: return False
        self._used.add(ds_id); return True
    @property
    def used(self): return len(self._used)

def rmse(y_true, y_pred):
    import numpy as np
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def score(acc_raw, datasets_used, nmax, lam=0.15):
    utility = 1.0 / (1.0 + acc_raw)
    penalty = lam * (datasets_used / max(1, nmax))
    return max(0.0, utility - penalty), utility, penalty

class GreenAgent:
    def __init__(self, context_dir="ecoagent/contexts/housing"):
        ctx = Path(context_dir)
        self.meta = yaml.safe_load((ctx/"meta.yaml").read_text())
        self.cat  = yaml.safe_load((ctx/"catalog.yaml").read_text())["datasets"]
        self.train = pd.read_csv(ctx/"train.csv")
        self.test  = pd.read_csv(ctx/"test.csv")
        self.budget = DatasetBudget(self.meta["budget"]["max_datasets"])
        self.target = self.meta["target"]
        self.horizon = int(self.meta["horizon"])
        self.leak_block = set(self.meta.get("leakage_guard_test_block", []))

    def describe(self):
        return {
            "context": self.meta["context_id"],
            "target": self.target,
            "horizon": self.horizon,
            "budget": {"max_datasets": self.budget.max},
            "catalog": [d["id"] for d in self.cat],
            "leakage_guard_test_block": list(self.leak_block)
        }

    def _columns_for(self, ds_id):
        entry = next(d for d in self.cat if d["id"] == ds_id)
        return entry["columns"]

    def request_dataset(self, ds_id: str, split="train"):
        if not self.budget.request(ds_id):
            raise RuntimeError(f"Dataset budget exceeded (max={self.budget.max})")

        if split == "test" and ds_id in self.leak_block:
            raise RuntimeError(f"Leakage guard: '{ds_id}' is not accessible on test split")

        cols = self._columns_for(ds_id)
        df = self.train if split == "train" else self.test
        for c in cols:
            if c not in df.columns:
                raise KeyError(f"Column '{c}' not found in {split} for dataset '{ds_id}'")
        return df[cols].dropna().copy()


    def _train_target_std(self):
        import numpy as np
        tgt_train = self.request_dataset("target", split="train")
        target_col = [c for c in tgt_train.columns if c != "Year"][0]
        vals = tgt_train[target_col].astype(float).values
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
        return max(std, 1e-8)

    def evaluate_predictions(self, y_pred):
        import numpy as np
        y_pred = np.array(y_pred, dtype=float)[: self.horizon]

        test = self.test.sort_values("Year")
        y_true = test[self.target].values[: self.horizon].astype(float)

        rmse_agent = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        scale = self._train_target_std()
        accuracy_norm = 1.0 / (1.0 + (rmse_agent / scale))

        eff_bonus = 1.0 - (self.budget.used / max(1, self.budget.max))

        scoring = self.meta.get("scoring", {})
        alpha = float(scoring.get("alpha_accuracy", 0.7))
        beta  = float(scoring.get("beta_efficiency", 0.3))

        final = max(0.0, min(1.0, alpha * accuracy_norm + beta * eff_bonus))

        return {
            "metric": "rmse",
            "rmse": round(rmse_agent, 5),
            "train_target_std": round(scale, 5),
            "accuracy_norm": round(accuracy_norm, 5),
            "datasets_used": self.budget.used,
            "eff_bonus": round(eff_bonus, 5),
            "final_score": round(final, 5),
        }
    
    def reset(self):
        self.budget = DatasetBudget(self.meta["budget"]["max_datasets"])