from sklearn.linear_model import LinearRegression

def run(ga):
    info = ga.describe()
    tgt_tr = ga.request_dataset("target", split="train")
    drv_tr = ga.request_dataset("median_listing_price", split="train")
    target_col = [c for c in tgt_tr.columns if c != "Year"][0]
    driver_col = [c for c in drv_tr.columns if c != "Year"][0]

    train = tgt_tr.merge(drv_tr, on="Year", how="inner").sort_values("Year")
    X = train[[driver_col]].values
    y = train[target_col].values
    model = LinearRegression().fit(X, y)

    drv_te = ga.request_dataset("median_listing_price", split="test").sort_values("Year").head(info["horizon"])
    y_pred = model.predict(drv_te[[driver_col]].values).tolist()
    return ga.evaluate_predictions(y_pred)