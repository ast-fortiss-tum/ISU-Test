from opensbt.visualization.isu_figures import boxplots, plot_metric_vs_time, statistics_table, last_values_table # , diversity_report,
import os


project = "OpenSBT-ISU"
path = f"./wandb_analysis/{project}"

os.makedirs(path, exist_ok=True)

statistics_table(project, "failures", os.path.join(path, "failures_stats.csv"))
statistics_table(project, "critical_ratio", os.path.join(path, "critical_ratio_stats.csv"))

plot_metric_vs_time(project, (18, 4), "failures", file_name=os.path.join(path, "failures_over_time.pdf"))
plot_metric_vs_time(project, (18, 4), "critical_ratio", file_name=os.path.join(path, "critical_ratio_over_time.pdf"))

last_values_table(project, ["failures", "critical_ratio"], path=os.path.join(path, "metrics.csv"))

boxplots(project, (18, 4), "failures", file_name=os.path.join(path, "failures_final.pdf"))
boxplots(project, (18, 4), "critical_ratio", file_name=os.path.join(path, "critical_ratio_final.pdf"))

# diversity_report(project, input=True, output_path=path)
# diversity_report(project, input=False, output_path=path)
