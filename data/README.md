# Data placement

Raw data is not committed. Unzip the organiser-provided `Student folder` so that the layout is:

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          # 80 trips
data/raw/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   # 10 trips
data/raw/evaluator_test/<trip_id>/...                                # Laptop 1 only, on evaluation day
```

The evaluator-test folder is never read by any development notebook (01–08). Only the API reads it,
and only the API's `scoring.py` opens `.route` files, after the prediction is finished.

`data/processed/` is created by `notebooks/02_build_dataset.ipynb`.
