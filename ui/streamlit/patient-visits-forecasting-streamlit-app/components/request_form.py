def build_timeseries_payload(data):
    return {"model_name":"AR","horizon":int(data["horizon"])}
def build_regression_payload(data):
    p={"model_name":"ELASTICNET"}; p.update(data); return p
