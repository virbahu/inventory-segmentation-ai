import numpy as np
def analyze_inv_segment(data):
    values = np.array([d["value"] for d in data])
    labels = [d["name"] for d in data]
    mu = np.mean(values); sig = np.std(values, ddof=1)
    cv = sig/max(mu, 1e-8)
    ranked = sorted(zip(labels, values), key=lambda x: -x[1])
    cumsum = np.cumsum([v for _,v in ranked])
    total = cumsum[-1] if len(cumsum) else 1
    categories = {{}}
    for i, (name, val) in enumerate(ranked):
        pct = cumsum[i]/total*100
        cat = "A" if pct <= 70 else "B" if pct <= 90 else "C"
        categories[name] = {{"value": round(val,2), "category": cat, "cum_pct": round(pct,1)}}
    return {{"mean": round(mu,2), "std": round(sig,2), "cv": round(cv,3), "categories": categories,
            "summary": {{"A": sum(1 for v in categories.values() if v["category"]=="A"),
                        "B": sum(1 for v in categories.values() if v["category"]=="B"),
                        "C": sum(1 for v in categories.values() if v["category"]=="C")}}}}
if __name__=="__main__":
    data = [{{"name": f"Item-{{i}}", "value": np.random.default_rng(42).exponential(5000, 1)[0]}} for i in range(100)]
    import json; print(json.dumps(analyze_inv_segment(data), indent=2))
