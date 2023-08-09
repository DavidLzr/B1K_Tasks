# B1K_Tasks
BEHAVIOR-1K tasks for robot learning projects

For each task, please replace the original `Rs_int_best.json` of the `Omnigibson` code base with our customized JSON file in each task folder. For example, in order to use the customized environments for the `putting_shoes` task:
```bash
cp putting_shoes/Rs_int_best.json $(path_to_Omnigibson)/omnigibson/data/og_dataset/scenes/Rs_int/json/Rs_int_best.json
```
