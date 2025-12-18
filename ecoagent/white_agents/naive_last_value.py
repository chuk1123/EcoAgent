def run(ga):
    info = ga.describe()
    tgt_tr = ga.request_dataset("target", split="train")
    target_col = [c for c in tgt_tr.columns if c != "Year"][0]
    last = tgt_tr[target_col].iloc[-1]
    y_pred = [last] * info["horizon"]
    return ga.evaluate_predictions(y_pred)